# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import cv2
from torch.nn.modules.utils import _pair, _quadruple

def load_model(model, checkpoint_file):
    if os.path.exists( checkpoint_file ):
        model_ = torch.nn.parallel.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model_.load_state_dict(torch.load( checkpoint_file ))
        model_ = model_.cuda()
        model_.eval()
    else:
        model_ = None
    return model_

def inv_depth(d):
    d = 1 / (d.clamp(min=1e-4))
    d[ d >= 1e3 ] = 0
    return d

def img2pc(pts, pose_c2w, K):
    """ 
    Project depth map to a 3D world-space point cloud.
    """
    pc = torch.matmul(pose_c2w, torch.stack( ((pts[0, ...] - K[0, 2]) * pts[2, ...] / K[0, 0],
                                              (pts[1, ...] - K[1, 2]) * pts[2, ...] / K[1, 1],
                                              -pts[2, ...],
                                              torch.ones_like(pts[2, ...], dtype=torch.double).to(pts.get_device() )), 0))
    return pc

def clean_depth_edges(dmap):
    """
    Remove any bands of incorrect depth that sometimes appear around edges in neural network-estimated depth maps, or
    when a depth map is filtered. These can create fly-away points when projected to 3D, and it's more efficient to
    remove them in image space rather than 3D. The filtering algorithm below is a bit hacky, but does the job.
    """
    h, w = dmap.shape

    out = dmap.clone()
    wsizes = [4] #[5, 4]

    for wsz in wsizes:
        dmap_ = out.reshape(h, -1, wsz).contiguous().permute(1, 0, 2)
        dmap_ = dmap_.reshape(dmap_.shape[0], -1, wsz, wsz).permute(1, 0, 2, 3).contiguous() #.view(-1, wsz * wsz)
        nh, nw = dmap_.shape[:2]
        dmap_ = dmap_.view( nh, nw, wsz * wsz )
        mn, mx = torch.min( dmap_, -1)[0].unsqueeze(-1), torch.max( dmap_, -1 )[0].unsqueeze(-1)
        dmap_ = (dmap_ - mn) / (mx - mn)
        dmap_[dmap_ < 0.5] = 0.
        dmap_[dmap_ > 0.5] = 1.
        dmap_ = dmap_ * (mx - mn) + mn
        dmap_ = dmap_.view(nh, nw, wsz, wsz )
        out = dmap_.permute(0, 2, 1, 3).contiguous().view(nh, wsz, -1).contiguous().view(h, w)

    idx = torch.isnan(out)
    out[idx] = dmap[idx]
    
    d = dmap.cpu().numpy()
    mn, mx = np.min(d), np.max(d)
    if mn == mx:
        return dmap
    d = (d - mn) / (mx - mn)
    d = (d * 255).astype(np.uint8)
    edges = cv2.Canny(d, 10, 20).astype(np.float32)/255
    
    dilation_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                        (dilation_size, dilation_size))
    edges = torch.tensor(cv2.dilate(edges, element)).bool()
    dmap[edges] = out[edges]

    return dmap

# InputPadder code from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/utils.py
class InputPadder:
    """ 
    Pads images such that dimensions are divisible by a specified number (default=8). 
    """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# MedianPool2d code from https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
class MedianPool2d(nn.Module):
    """ 
    Median pool (usable as median filter when stride=1) module.
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=1, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        
    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

def scale_tiled(src, tgt):
    """ 
    Scale and shift tiles of src depth to match corresponding tiles in tgt.
    This function is used for bringing two depth maps from a monocular estimator 
    into a common reference scale.
    """

    device = src.get_device()
    if device == -1:
        device = 'cpu'
        
    mask = torch.logical_and(tgt > 0, src > 0)

    _, h, w = tgt.shape
    tile_sz = w // 4
    tile_overlap = tile_sz // 12

    src_inv = inv_depth(src) 
    tgt_inv = inv_depth(tgt)

    sx = np.arange(0, w, tile_sz - tile_overlap)
    sy = np.arange(0, h, tile_sz - tile_overlap)

    # Scale and shift factors for the (sy, sx) tile grid
    S = torch.zeros( (len(sy), len(sx)) ).to(device)
    T = torch.zeros( (len(sy), len(sx)) ).to(device)

    for yi, y in enumerate(sy):
        for xi, x in enumerate(sx):
            tile_src = src_inv[:, y:y+tile_sz, x:x+tile_sz]
            tile_tgt = tgt_inv[:, y:y+tile_sz, x:x+tile_sz]
            tile_msk = mask[:, y:y+tile_sz, x:x+tile_sz]
            _, s, t = scale_depth(tile_src, tile_tgt, tile_msk, device)
            S[yi, xi] = s
            T[yi, xi] = t

    # Upsample the overlapping tile factors to the image size
    # The tile overlap, along with the bilinear upsampling ensures there are no sudden changes
    # in scale and shift between neighboring tile regions in the final result
    S_ = F.interpolate( S.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    T_ = F.interpolate( T.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    
    src_inv_scaled = src_inv * S_.squeeze(0) + T_.squeeze(0)
    src_scaled = inv_depth(src_inv_scaled)
    
    return src_scaled

def scale_depth(prediction, target, mask, device='cpu'):
    # Remove outliers as they will  skew the least squares optimization process
    mn_pred, mx_pred = torch.quantile(prediction, 0.01), torch.quantile(prediction, 0.99)
    mn_target, mx_target = torch.quantile(target, 0.01), torch.quantile(target, 0.99)
    mask_pred = torch.logical_and(prediction >= mn_pred, prediction <= mx_pred)
    mask_target = torch.logical_and(target >= mn_target, target <= mx_target)
    mask = mask * mask_pred.float() * mask_target.float()
    
    scale, shift = compute_scale_and_shift(prediction, target, mask, device)
    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

    return prediction_aligned, scale, shift


def compute_scale_and_shift(prediction, target, mask, device='cpu'):
    """
    Perform a least squares optimization to compute a scale and shift factor that 
    aligns the prediction and target (inverse) depths. The code is adapted from
    https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022

    A good explanation of the process can be found in the paper:
    " Towards Robust Monocular Depth Estimation"
    """
    
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    #if not valid:
    a_00 = a_00 + 0.1
    a_11 = a_11 + 0.1
    det = a_00 * a_11 - a_01 * a_01
        
    x_0 = (a_11 * b_0 - a_01 * b_1) / det
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    if(torch.abs(x_0) <= 1e-5):
        if(torch.abs(x_1) <= 1e-5):
            x_0 = torch.ones_like(x_0).to(device)
            x_1 = torch.zeros_like(x_1).to(device)
    
    h, w = prediction.shape[1:]
    if a_11 < h * w * 0.5:
        x_0 = torch.ones_like(x_0).to(device)
        x_1 = torch.zeros_like(x_1).to(device)

    return x_0, x_1

