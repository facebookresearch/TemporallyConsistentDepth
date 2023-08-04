import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import TemporalFusionNet 
from networks import SpatialFusionNet
from pcrender import Splat, Match
from utils import utils
import config

class PCUpdate:

    def __init__(self, h, w, device='cpu'):
        super(PCUpdate, self).__init__()

        self.device = device
        self.h, self.w = h, w
        self.ygrid, self.xgrid = torch.meshgrid(torch.arange(h - 1, -1, -1), torch.arange(0, w), indexing="ij")
        self.pc, self.pc_w = [], []
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.medianpool = utils.MedianPool2d()
        self.match = Match(device)
        self.splat = Splat(device)

        # Initialize Temporal Fusion and Spatial Fusion Networks
        self.temp_fusion_net = utils.load_model( TemporalFusionNet(),
                                                 os.path.join(os.getcwd(), 'weights/tcod_temporal_fusion_net.pt') )
        self.spatial_fusion_net = utils.load_model( SpatialFusionNet(),
                                                    os.path.join(os.getcwd(), 'weights/tcod_spatial_fusion_net.pt') )
        if self.temp_fusion_net == None or self.spatial_fusion_net == None:
            print("Could not find weights for Spatial Fusion or Temporal Fusion Network. Exiting...")
            exit()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True    

        self.global_idx = 0
        self.fused_depth = None
        
    def points(self):
        return self.pc, self.pc_w

    def clear(self):
        self.pc_w = []
        self.pc = []


    def init_pc(self, dmap_obv, poses_w2c, K, rgb):
        pc, pc_w = [], []
        for i in range(dmap_obv.shape[0]):
            pts_obv = utils.img2pc( torch.stack( (self.xgrid.to(self.device).unsqueeze(0), 
                                                  self.ygrid.to(self.device).unsqueeze(0), 
                                                  dmap_obv[i, ...].to(self.device)), 0 ).view(3, -1), 
                                    torch.linalg.inv(poses_w2c[i, ...]), K[i, ...]).float()
            pts_obv = torch.cat( (pts_obv, rgb[i, ...].reshape(3, -1)), 0)
            pc.append(pts_obv)
            pc_w.append( torch.ones( pts_obv.shape[-1] ).to(self.device) * 1e-4)
        self.pc = pc
        self.pc_w = pc_w

    def batch_pts(self):
        # Pad all point clouds so that they are the same size for batching
        max_pts = max([ pc.shape[-1] for pc in self.pc ])
        pc_padded = torch.stack([ torch.cat( (pc, torch.ones((pc.shape[0], max_pts - pc.shape[-1])).to(self.device)), -1 )
                                  for pc in self.pc], 0)
        pc_w_padded = torch.stack([ torch.cat( (pc_w, torch.zeros((max_pts - pc_w.shape[-1],)).to(self.device)), -1 )
                                    for pc_w in self.pc_w], 0)
        return pc_padded, pc_w_padded

    def temporal_fusion(self, obv_depth, poses_w2c, K, obv_rgb):
        pc, pc_w = self.batch_pts()

        if self.global_idx <= 0: # Very first frame, keep all points with confidence of one
            prior_depth, prior_conf, prior_rgb = self.splat( pc, pc_w, poses_w2c, K, self.h, self.w, zfar=config.zfar )
            prior_conf = torch.ones_like(prior_conf)
        else:
            prior_depth, prior_conf, prior_rgb = self.splat( pc, pc_w, poses_w2c, K, self.h, self.w, zfar=config.zfar,
                                                             conf_thresh=config.tstable/config.max_w)
        # Outlier filtering and smoothing
        prior_med = self.medianpool( torch.cat( (prior_rgb, prior_depth, prior_conf), 1) )
        prior_depth_med = prior_med[:, 3:4, ...]
        prior_rgb_med = prior_med[:, :3, ...]
        prior_conf_med = prior_med[:, -1:, ...]
        outlier_mask = ( torch.abs(prior_depth_med - prior_depth) ) > 5e-2
        prior_depth[outlier_mask] = prior_depth_med[outlier_mask]
        prior_conf = prior_conf_med
        prior_rgb[ outlier_mask.expand(-1, 3, -1, -1) ] = prior_rgb_med[ outlier_mask.expand(-1, 3, -1, -1) ]
        
        padder = utils.InputPadder( obv_depth.shape, divis_by=16)
        obv_rgb_pad, prior_rgb_pad, obv_inv_depth_pad, prior_inv_depth_pad = padder.pad(obv_rgb,
                                                                                        prior_rgb,
                                                                                        utils.inv_depth(obv_depth),
                                                                                        utils.inv_depth(prior_depth))

        # Generate a blending mask for the prior and observed depths. Ideally, alpha should only identify dynamic changes
        # in the scene from prior to observed. This allows moving objects to be incorporated into the prior depth, without
        # affecting the static regions, which remain consistent with previous frames. Any errors in the static regions are
        # corrected in the next stage of the pipeline. 
        alpha = self.temp_fusion_net( obv_inv_depth_pad,
                                      obv_rgb_pad,
                                      prior_inv_depth_pad,
                                      prior_rgb_pad )
        alpha = padder.unpad( alpha )
        alpha[prior_depth <= 0] = 0

        # Scale the obvserved depth map using the static regions from the prior depth, this is useful for monocular depth method
        obv_depth_affine_scaled = utils.scale_tiled( obv_depth.squeeze(0), (prior_depth * (alpha > 0.5)).squeeze(0) ).unsqueeze(0)
        fused_depth = prior_depth * alpha + obv_depth_affine_scaled * (1 - alpha)
        prior_conf *= alpha

        # Filter the confidence as, coming from splatted points, it can become quite noisy
        prior_conf_down = F.interpolate( prior_conf, scale_factor=0.125, mode='bilinear')
        prior_conf = F.interpolate( prior_conf_down, alpha.shape[-2:], mode='bilinear')

        return fused_depth, prior_conf, alpha
    
    def spatial_fusion(self, prior_depth, obv_depth, obv_rgb, prior_conf):
        padder = utils.InputPadder( obv_depth.shape, divis_by=16)
        obv_rgb_pad, obv_inv_depth_pad, prior_inv_depth_pad, prior_conf_pad = padder.pad(obv_rgb,
                                                                                 utils.inv_depth(obv_depth),
                                                                                 utils.inv_depth(prior_depth),
                                                                                 prior_conf)

        pred_unc = padder.unpad( self.spatial_fusion_net( torch.cat( (prior_inv_depth_pad, obv_rgb_pad), 1) ))
        pred_conf = torch.exp(-1.0 * pred_unc)
        obv_conf = padder.unpad( torch.exp(-1.0 * self.spatial_fusion_net( torch.cat( (obv_inv_depth_pad, obv_rgb_pad), 1) )))

        scaled_pred_conf = pred_conf * prior_conf
        fused_depth = (prior_depth * scaled_pred_conf + obv_conf * obv_depth) / (scaled_pred_conf + obv_conf)

        return fused_depth, scaled_pred_conf, obv_conf 

    
    def update(self, obv_rgb, obv_depth, poses_w2c, K ):
        if len(self.pc) <= 0:
            self.init_pc( obv_depth, poses_w2c, K, obv_rgb) 
            self.fused_depth = obv_depth.clone().detach()
        
        b, _, h, w = obv_depth.shape

        with torch.no_grad():

            #-----------------#
            # TEMPORAL FUSION #
            #-----------------#
            prior_depth, prior_conf, alpha = self.temporal_fusion( obv_depth, poses_w2c, K, obv_rgb )

            #----------------#
            # SPATIAL FUSION #
            #----------------#
            fused_depth, pred_conf, obv_conf = self.spatial_fusion(prior_depth, obv_depth, obv_rgb, prior_conf) 
            self.fused_depth = fused_depth.clone().detach()

            #--------------------#
            # POINT-BASED FUSION #
            #--------------------#
            # Match point cloud to dmap
            pc_batched, pc_w_batched = self.batch_pts()
            obv_match_mask, prior_match_idxmap, prior_conf = self.match(alpha > 0.5,
                                                                        pred_conf,
                                                                        pc_batched,
                                                                        pc_w_batched / config.max_w,
                                                                        poses_w2c,
                                                                        K, self.h, self.w)
            for i in range(b): 
                pc = pc_batched[i, ...]
                pc_w = pc_w_batched[i, ...]

                prior_match_idx = prior_match_idxmap[i, ...].flatten()
                prior_match_idx = prior_match_idx[prior_match_idx >= 0]
                
                # Blend matching points from observed depth map with global point cloud
                prior_match_pts = pc[:, prior_match_idx]
                prior_match_conf = prior_conf[i, 0, prior_match_idx] 
                
                obv_pts = utils.img2pc( torch.stack( (self.xgrid.to(self.device).unsqueeze(0), 
                                                      self.ygrid.to(self.device).unsqueeze(0), 
                                                      obv_depth[i, ...].to(self.device)), 0 ).view(3, -1),
                                        torch.linalg.inv(poses_w2c[i, ...]), K[i, ...]).float()
                obv_pts = torch.cat( (obv_pts, obv_rgb[i, ...].reshape(3, -1)), 0)

                obv_match_pts = obv_pts[:, obv_match_mask[i, ...].flatten().bool()]
                obv_match_conf = obv_conf[i, ...].reshape(1, -1)[:, obv_match_mask[i, ...].flatten().bool()] 
                
                fused_pts = (prior_match_conf * prior_match_pts + obv_match_conf * obv_match_pts) / (prior_match_conf + obv_match_conf) 
                
                pc[:, prior_match_idx] = fused_pts
                pc_w -= 1 # Points that are not matched will decrease in confidence over frames
                pc_w[prior_match_idx] = prior_conf[i, 0, prior_match_idx]
                pc_w[prior_match_idx] += obv_match_conf.squeeze(0)
                pc_w = torch.clamp(pc_w, 0, config.max_w) 
                
                # Remove points with weight <= 0
                pc = pc[:, (pc_w > 0).flatten()]
                pc_w = pc_w[(pc_w > 0).flatten()]
                
                # Add unmatched points from tgt to the point cloud
                obv_unmatch_pts = obv_pts[:, torch.logical_not(obv_match_mask[i, ...].bool()).flatten()]
                obv_unmatch_conf = obv_conf[i, ...].reshape(1, -1)[:, torch.logical_not(obv_match_mask[i, ...].flatten().bool())]
                
                pc = torch.cat( (pc, obv_unmatch_pts), -1 )
                pc_w = torch.cat( (pc_w, obv_unmatch_conf.squeeze(0) ), -1)
                
                self.pc[i] = pc
                self.pc_w[i] = pc_w

        self.global_idx += 1


    
