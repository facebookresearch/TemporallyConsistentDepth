# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

RAFT_OPTICAL_FLOW_PATH = '' # Define the path to the RAFT Optical Flow code

import os
import torch
import numpy as np
from ssim import SSIM
import argparse
import sys
sys.path.append( os.path.join(RAFT_OPTICAL_FLOW_PATH, 'core') )
sys.path.appand( '../utils' )
from raft import RAFT
from utils.utils import InputPadder
import torch.nn.functional as F
from skimage.metrics import structural_similarity

class Metrics():

    def __init__(self, h, w, device):
        super().__init__()
        
        # Declare optical flow object
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()

        self.device = device

        raft_oflow = torch.nn.DataParallel(RAFT(args))
        raft_oflow.load_state_dict(torch.load( os.path.join( RAFT_OPTICAL_FLOW_PATH, 'models/raft-things.pth') ))

        self.raft_oflow = raft_oflow.module
        self.raft_oflow.to(self.device)
        self.raft_oflow.eval()

        self.h = h
        self.w = w
        self.ygrid, self.xgrid = torch.meshgrid(torch.arange(h - 1, -1, -1), torch.arange(0, w), indexing="ij")
        self.xgrid = self.xgrid.to(device)
        self.ygrid = self.ygrid.to(device)
        
        self.flow_limit = 250 # pixels
        
    def oflow(self, image1, image2):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            flow_low, flow_up = self.raft_oflow(image1.contiguous(), image2.contiguous(), iters=20, test_mode=True)
            
        flow_up = padder.unpad(flow_up)
        return flow_up
    
    def TCC(self, d0, d1, gt0, gt1, mask=None):
        if mask == None:
            mask = torch.ones_like(d0).to(d0.get_device())

        ssimloss = SSIM(1.0, nonnegative_ssim=True)
        return  ssimloss( (torch.abs(d1 - d0) * mask.float()).expand(-1, 3, -1, -1),
                          (torch.abs(gt1 - gt0) * mask.float()).expand(-1, 3, -1, -1) )

        
    def TCM(self, d0, d1, gt0, gt1, mask=None):
        if mask == None:
            mask = torch.ones_like(d0).to(d0.get_device())

        b, _, h, w = d0.shape
        ssimloss = SSIM(1.0, nonnegative_ssim=True, size_average=False)

        dmax = torch.max(gt0.view(b, -1), -1)[0].view(b, 1, 1, 1).expand(-1, 3, -1, -1)
        dmin = torch.min(gt0.view(b, -1), -1)[0].view(b, 1, 1, 1).expand(-1, 3, -1, -1)
        
        d0_ = (d0.expand(-1, 3, -1, -1).to(self.device) - dmin) / (dmax - dmin) * 255.
        d1_ = (d1.expand(-1, 3, -1, -1).to(self.device) - dmin) / (dmax - dmin) * 255.
        flow = self.oflow( d0_, d1_ )

        gt0_ = (gt0.expand(-1, 3, -1, -1).to(self.device) - dmin) / (dmax - dmin) * 255.
        gt1_ = (gt1.expand(-1, 3, -1, -1).to(self.device) - dmin) / (dmax - dmin) * 255.
        flow_gt = self.oflow( gt0_, gt1_ )
        flow_mask = torch.sum(flow > self.flow_limit, 1, keepdim=True) == 0

        mask = torch.logical_and(flow_mask, mask)
        
        ssim = torch.mean(ssimloss( torch.cat( (flow, torch.ones_like(flow[:, 0, None, ...])), 1) * mask.expand(-1, 3, -1, -1),
                                    torch.cat( (flow_gt, torch.ones_like(flow[:, 0, None, ...])), 1) * mask.expand(-1, 3, -1, -1) )[:, :2])
        return ssim

        
    def rTC(self, d0, d1, img0, img1, resample=False, pose_d0_to_d1=None, K=None, flow=None, mask=None):
        if mask == None:
            mask = torch.ones_like(d0).to(d0.get_device())

        if flow is None:
            flow = self.oflow(img0, img1)
        flow_mask = torch.sum(flow > self.flow_limit, 1, keepdim=True) == 0

        if resample:
            # Transform d0 to camera space depth in frame of d1
            pts =  torch.stack( (self.xgrid.to(self.device),
                                 self.ygrid.to(self.device),
                                 d0[0, 0, ...].to(self.device)), 0 ).view(3, -1)
            pc = torch.matmul(pose_d0_to_d1, torch.stack( ((pts[0, ...] - K[0, 2]) * pts[2, ...] / K[0, 0],
                                                           (pts[1, ...] - K[1, 2]) * pts[2, ...] / K[1, 1],
                                                           -pts[2, ...],
                                                           torch.ones_like(pts[2, ...], dtype=torch.double).to(self.device)), 0))
            d0_t = torch.abs(pc[2, ...].view(self.h, self.w)).view(1, 1, self.h, self.w)
        else:
            d0_t = d0

        # Use flow to sample points from d1. This is the camera space depth in d1
        x_norm = (self.xgrid + flow[0, 0, ...]) / float(self.w - 1) * 2 - 1.0 
        y_norm = (self.ygrid - flow[0, 1, ...]) / float(self.h - 1) * 2 - 1.0  # TODO: Does float y increase up or down
        y_norm *= -1
        grid = torch.stack( (x_norm, y_norm), -1).float().unsqueeze(0)
        d1_sampled = F.grid_sample( d1, grid, align_corners=True)
        img1_sampled = F.grid_sample( img1 / 255., grid, align_corners=True)

        # Compare transformed and sampled points
        mask = torch.exp(-50. * torch.sqrt(torch.sum((img0/255. - img1_sampled)**2, dim=1, keepdim=True))) * flow_mask * mask * (d1_sampled > 0) > 1e-2
        m = torch.sum(mask)

        tau = 1.01
        
        x1 = d0_t / d1_sampled
        x2 = d1_sampled / d0_t
        
        x1[ torch.isinf(x1) ] = -1e10
        x2[ torch.isinf(x2) ] = -1e10
        x = torch.max(torch.cat( (x1, x2), 1 ), 1)[0] < tau

        err = torch.sum(x * mask) / m
        return err


    def OPW(self, d0, d1, img0, img1, resample=False, pose_d0_to_d1=None, K=None, flow=None, mask=None):
        if mask == None:
            mask = torch.ones_like(d0).to(d0.get_device())

        if flow is None:
            flow = self.oflow(img0, img1)
        flow_mask = torch.sum(flow > self.flow_limit, 1, keepdim=True) == 0
        
        if resample:
            # Transform d0 to camera space depth in frame of d1
            pts =  torch.stack( (self.xgrid, self.ygrid, d0[0, 0, ...]), 0 ).view(3, -1)
            pc = torch.matmul(pose_d0_to_d1, torch.stack( ((pts[0, ...] - K[0, 2]) * pts[2, ...] / K[0, 0],
                                                           (pts[1, ...] - K[1, 2]) * pts[2, ...] / K[1, 1],
                                                           -pts[2, ...],
                                                           torch.ones_like(pts[2, ...], dtype=torch.double).to(self.device)), 0))
            d0_t = torch.abs(pc[2, ...].view(self.h, self.w)).view(1, 1, self.h, self.w)
        else:
            d0_t = d0
        
        # Use flow to sample points from d1. This is the camera space depth in d1
        x_norm = (self.xgrid + flow[0, 0, ...]) / float(self.w - 1) * 2 - 1.0 
        y_norm = (self.ygrid - flow[0, 1, ...]) / float(self.h - 1) * 2 - 1.0  # TODO: Does float y increase up or down
        y_norm *= -1
        grid = torch.stack( (x_norm, y_norm), -1).float().unsqueeze(0)
        d1_sampled = F.grid_sample( d1, grid, align_corners=True)
        img1_sampled = F.grid_sample( img1 / 255., grid, align_corners=True)

        # Compare transformed and sampled points
        mask = torch.exp(-50. * torch.sqrt(((img0/255. - img1_sampled)**2).sum(1))) * flow_mask * mask * (d1_sampled > 0) > 1e-2
        m = torch.sum(mask)
        
        err = torch.sum(torch.abs(d1_sampled - d0_t) * mask) / m
        return err


    def self_consistency(self, d0, d1, img0, img1, pose_d0_to_d1, K, resample=False, mask=None, flow=None):
        if mask == None:
            mask = torch.ones_like(d0).to(d0.get_device())

        _, _, h, w = d0.shape
        
        # Transform d0 to camera space depth in frame of d1
        pts =  torch.stack( (self.xgrid.to(self.device),
                             self.ygrid.to(self.device),
                             d0[0, 0, ...].to(self.device)), 0 ).view(3, -1)
        pc = torch.matmul(pose_d0_to_d1, torch.stack( ((pts[0, ...] - K[0, 2]) * pts[2, ...] / K[0, 0],
                                                       (pts[1, ...] - K[1, 2]) * pts[2, ...] / K[1, 1],
                                                       -pts[2, ...],
                                                       torch.ones_like(pts[2, ...], dtype=torch.double).to(self.device)), 0))
        pc[2, ...] = torch.abs(pc[2, ...])

        xt = pc[0, ...] * K[0, 0] / pc[2, ...] + K[0, 2] 
        yt = pc[1, ...] * K[1, 1] / pc[2, ...] + K[1, 2] 
        
        flowx = xt.view(h, w) - self.xgrid.to(self.device)
        flowy = -(yt.view(h, w) - self.ygrid.to(self.device))
        flow = torch.stack( (flowx, flowy), 0 )
        
        if resample:
            d0_t = pc[2, ...].view(self.h, self.w).view(1, 1, self.h, self.w)
        else:
            d0_t = d0

        # Use flow to sample points from d1. This is the camera space depth in d1
        x_norm = (self.xgrid + flow[0, ...]) / float(self.w - 1) * 2 - 1.0 
        y_norm = (self.ygrid - flow[1, ...]) / float(self.h - 1) * 2 - 1.0  # TODO: Does float y increase up or down
        y_norm *= -1
        grid = torch.stack( (x_norm, y_norm), -1).float().unsqueeze(0)
        d1_sampled = F.grid_sample( d1, grid, align_corners=True)

        img1_sampled = F.grid_sample( img1, grid, align_corners=True) / 255.

        # Compare transformed and sampled points
        mask = torch.exp(-50. * torch.sqrt(((img0/255. - img1_sampled)**2).sum(1))) * mask * (d1_sampled > 0) > 1e-2
        m = torch.sum(mask)
        
        err = torch.sum(torch.abs(d1_sampled - d0_t) * mask) / m
        return err

    def stderr(self, din, dgt, mask=None):
        if mask == None:
            mask = torch.ones_like(din).to(din.get_device())
        # din: b, n, 1, h, w
        n, _, _, _  = din.shape
        m = torch.sum( (mask > 0).view(n, -1), -1)
        err = torch.sum(torch.abs(din * mask - dgt * mask).view(n, -1), -1) / m 
        stderr = torch.std( err, -1 )
        return stderr

    def rms(self, din, dgt, mask=None):
        if mask == None:
            mask = torch.ones_like(din).to(din.get_device())
        n = torch.sum(mask > 0)
        err = torch.sqrt(torch.sum( torch.square(din - dgt) * mask.float() )/n)
        return err

    def rel(self, din, dgt, mask=None):
        if mask == None:
            mask = torch.ones_like(din).to(din.get_device())
        n = torch.sum(mask > 0)
        err = torch.sum(torch.abs(din - dgt) * mask.float() / dgt.clamp(min=1)) / n
        return err
    
    def rms_log(self, din, dgt, mask=None):
        if mask == None:
            mask = torch.ones_like(din).to(din.get_device())
        n = torch.sum(mask > 0)
        err = torch.sqrt(torch.sum( torch.square(torch.log(din * mask + 1e-10) - torch.log(dgt * mask + 1e-10)) )/n)
        return err

    def bad_pixels(self, din, dgt, tau, mask=None):
        if mask == None:
            mask = torch.ones_like(din).to(din.get_device())

        m = torch.sum(mask)
        
        x1 = din / dgt
        x2 = dgt / din
        
        x1[ torch.isinf(x1) ] = -1e10
        x2[ torch.isinf(x2) ] = -1e10
        x = torch.max(torch.cat( (x1, x2), 1 ), 1)[0] < tau

        err = torch.sum(x * mask) / m
        return err



