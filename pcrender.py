# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
import numpy as np

class Match(torch.nn.Module):
    def __init__(self, device='cuda:0'):
        super(Match, self).__init__()
        self.wsz = 2 # Super-sampling factor
        self.device = device

    def forward(self, alpha, beta, pp, conf, pose_w2c, K, h, w, znear=0.1, zfar=1e2):

        # pts is b x 4 x n
        b, _, n = pp.shape
        h, w = alpha.shape[-2:]
        
        pc = torch.matmul(pose_w2c, pp[:, :4, ...].double())

        xc = (pc[:, 0, ...] * K[:, 0, 0].view(-1, 1)) / torch.abs(pc[:, 2, ...]) + K[:, 0, 2].view(-1, 1).float()
        yc = (pc[:, 1, ...] * K[:, 1, 1].view(-1, 1)) / torch.abs(pc[:, 2, ...]) + K[:, 1, 2].view(-1, 1).float()
        z = torch.abs(pc[:, 2, ...] )
        rgb = pp[:, 4:7, ...]

        # Sample weights of projected points
        x_norm = xc / float(w - 1) * 2 - 1.0 
        y_norm = yc / float(h - 1) * 2 - 1.0 
        y_norm *= -1
        grid = torch.stack( (x_norm, y_norm), -1).float().view(-1, 1, n, 2)
        conf_sampled = F.grid_sample( beta, grid, align_corners=True).view(-1, 1, n) 
        
        # Super sampling
        wsz = self.wsz
        xc = xc * wsz
        yc = yc * wsz
        alpha_ = torch.flip(torch.repeat_interleave(torch.repeat_interleave(alpha, wsz, dim=-1), wsz, dim=-2), [-2])

        x = torch.round(xc).long()
        y = torch.round(yc).long()

        out_of_bounds = torch.logical_or( torch.logical_or(torch.logical_or(torch.logical_or(x < 0, x >= w * wsz), torch.logical_or(y < 0, y >= h * wsz)),
                                                           torch.logical_or(z < znear, z > zfar)), conf <= 0 )
        not_out_of_bounds = torch.logical_not(out_of_bounds)

        match, match_src_idx = [], []
        pool = torch.nn.MaxPool2d(wsz, stride=wsz, padding=0 , return_indices=True)
        
        for bidx in range(b):

            idx = torch.linspace(0, n - 1, n).long().to(self.device)
            cmap_hi = torch.zeros( h * wsz, w * wsz ).to(self.device) 
            idx_hi = torch.ones( h * wsz, w * wsz ).to(self.device).long() * -1
            m = not_out_of_bounds[bidx, ...] 

            cmap_hi[y[bidx, m], x[bidx, m]] = conf[bidx, m].float()
            idx_hi[y[bidx, m], x[bidx, m]] = idx[m]

            keep_idx = torch.logical_and((alpha_[bidx, 0, ...] > 0), (cmap_hi > 0))

            # Select the point with the highest confidence from amonst the valid points
            cmap_, cmax_indices = pool( (cmap_hi * keep_idx).unsqueeze(0).unsqueeze(0) )
            idx_hi[~keep_idx] = -1
            idx_ = idx_hi.view(1, -1)[:, cmax_indices].view(h, w)

            match.append( torch.flip(pool(keep_idx.unsqueeze(0).unsqueeze(0).float())[0].view(h, w), [0]) )
            match_src_idx.append(torch.flip(idx_, [0]))
            
        return torch.stack(match, 0).unsqueeze(1), torch.stack(match_src_idx, 0).unsqueeze(1), conf_sampled

    
class Splat(torch.nn.Module):
    def __init__(self, device='cuda:0'):
        super(Splat, self).__init__()
        self.wsz = 1 # Super-sampling factor
        self.device = device

        self.gap_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.gap_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[0, 0, 0],[-1, 2., -1],[0, 0., 0]])).float().unsqueeze(0).unsqueeze(0))
        self.gap_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.gap_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[0, -1., 0],[0, 2., 0],[0, -1., 0]])).float().unsqueeze(0).unsqueeze(0))
        self.gap_x = self.gap_x.to(device)
        self.gap_y = self.gap_y.to(device)

        self.grad_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.grad_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[0, 0, 0],[-1, 0., 1],[0, 0., 0]])).float().unsqueeze(0).unsqueeze(0))
        self.grad_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.grad_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[0, -1., 0],[0, 0., 0],[0, 1., 0]])).float().unsqueeze(0).unsqueeze(0))
        self.grad_x = self.grad_x.to(device)
        self.grad_y = self.grad_y.to(device)

        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.filters = torch.tensor([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 1],
                                     [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 1, 1, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 1, 1],
                                     [0, 1, 1, 0, 0, 1, 0, 0, 0],
                                     [1, 1, 0, 1, 0, 0, 0, 0, 0]]).view(8, 1, 3, 3).cuda().float()
        
    def forward(self, pp, conf, pose_w2c, K, h, w, znear=0.1, zfar=1e3, conf_thresh=0):

        # pts is b x (4+3) x n
        b, _, n = pp.shape
        
        pc = torch.matmul(pose_w2c, pp[:, :4, ...].double())

        xc = (pc[:, 0, ...] * K[:, 0, 0].view(-1, 1)) / torch.abs(pc[:, 2, ...]) + K[:, 0, 2].view(-1, 1).float()
        yc = (pc[:, 1, ...] * K[:, 1, 1].view(-1, 1)) / torch.abs(pc[:, 2, ...]) + K[:, 1, 2].view(-1, 1).float()
        z = torch.abs(pc[:, 2, ...] )
        
        # Super sampling
        wsz = self.wsz
        xc = xc * wsz
        yc = yc * wsz
        
        x = torch.round(xc).long()
        y = torch.round(yc).long()
        
        out_of_bounds = torch.logical_or( torch.logical_or(torch.logical_or(torch.logical_or(x < 0, x >= w * wsz), torch.logical_or(y < 0, y >= h * wsz)),
                                                           torch.logical_or(z < znear, z > zfar)), conf <= conf_thresh )

        z[out_of_bounds] = 1e-10
        dmin = torch.min( 1/z, dim=-1)[0].view(-1, 1)
        z[out_of_bounds] = 1e10
        dmax = torch.max( 1/z, dim=-1)[0].view(-1, 1)

        nq = 64
        do = ((1/z - dmin) / (dmax - dmin) * (nq - 1)).int() # depth ordinal: used for masking
        do[out_of_bounds] = -1
        
        depths = []
        confs = []
        rgbs = []
        pool = torch.nn.MaxPool2d(wsz, stride=wsz, padding=0, return_indices=True)
        
        for bidx in range(b):
            dmap = torch.zeros( nq, h, w ).to(self.device)
            cmap = torch.zeros( nq, h, w ).to(self.device)
            rgb = torch.zeros( nq, 3, h, w ).to(self.device)
            for i in range(nq):

                dmap_hi = torch.zeros( h * wsz, w * wsz ).to(self.device)
                cmap_hi = torch.ones( h * wsz, w * wsz ).to(self.device) * -1
                rgb_hi = torch.zeros( 3, h * wsz, w * wsz ).to(self.device)
                
                m = do[bidx, ...] == i
                xi, yi = x[bidx, m], y[bidx, m]
                
                cmap_hi[yi, xi] = conf[bidx, m].float()
                dmap_hi[yi, xi] = z[bidx, m].float()
                rgb_hi[:, yi, xi] = pp[bidx, 4:, m].float()
                
                # Select the point with the maximum confidence in the window
                cmap_, cmap_indices = pool( cmap_hi.unsqueeze(0).unsqueeze(0) )
                dmap_ = dmap_hi.view(1, -1)[:, cmap_indices.view(h, w)].view(h, w)
                rgb_ = rgb_hi.view(3, -1)[:, cmap_indices.view(h, w)].view(3, h, w)
                
                dmap[i, ...] = dmap_
                cmap_[cmap_ < 0] = 0.
                cmap[i, ...] = cmap_
                rgb[i, ...] = rgb_
                
            # Min along depth dimension to get flattened depth map
            dmap[dmap <= 0] = 1e10
            dmap_flat, min_idx = torch.min(dmap, 0)
            dmap_flat[dmap_flat >= 1e10] = 0

            cmap_flat = torch.gather( cmap, 0, min_idx.unsqueeze(0)).squeeze(0)
            rgb_flat = torch.gather( rgb, 0, min_idx.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)).squeeze(0)

            dmap_flat = torch.flip(dmap_flat, [0])
            cmap_flat = torch.flip(cmap_flat, [0])
            rgb_flat = torch.flip(rgb_flat, [1])
            
            depths.append(dmap_flat)
            confs.append(cmap_flat)
            rgbs.append(rgb_flat)

        depths = torch.stack(depths, 0).view(-1, 1, h, w)
        confs = torch.stack(confs, 0).view(-1, 1, h, w)
        rgbs = torch.stack(rgbs, 0).view(-1, 3, h, w)

        for i in range(2):
            # Identify background pixels where some pixels in a 3x3 window are foreground pixels
            tofill = torch.logical_and(F.conv2d( depths, torch.ones((3, 3)).view(1, 1, 3, 3).cuda(), padding=1) > 0, depths <= 0)
            # Identify pixels that must be filled based on Rosenthal and Linsen
            out = F.conv2d(depths, self.filters, padding=1)
            out = (torch.abs(torch.prod(out, 1)) > 1e-10).unsqueeze(1) * tofill

            pooled_depths = self.pool(depths)
            pooled_rgbs = self.pool(rgbs)
            pooled_confs = self.pool(confs)
            
            depths = depths * (1 - out.float()) + out.float() * pooled_depths
            confs = confs * (1 - out.float()) + out.float() * pooled_confs
            out = out.expand(-1, 3, -1, -1)
            rgbs = rgbs * (1 - out.float()) + out.float() * pooled_rgbs

        return (depths, confs, rgbs)
