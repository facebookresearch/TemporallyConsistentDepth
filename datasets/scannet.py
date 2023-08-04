# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import torch
from torch.utils.data import Dataset
from dpt_wrapper import DPTWrapper
import torch.nn.functional as F
import numpy as np
import glob
import cv2
import utils.imutils as imutils
import utils.utils as utils

class ScanNetDataset(Dataset):
    def __init__(self, rootdir, output_height=484, output_width=684):
        super(ScanNetDataset, self).__init__()

        self.rootdir = rootdir
        self.ho, self.wo = output_height, output_width

        # Replace the following line with any desired monocular depth estimator.
        # Our method has been tested with DPT, and uses it by default. 
        self.monocular_depth = DPTWrapper(model_path='./DPT/weights/dpt_hybrid-midas-501f0c75.pt')
            
        self.image_paths = glob.glob(self.rootdir + "/color/*.jpg")

        h, w = imutils.png2np( self.image_paths[0] ).shape[:2]
        sh, sw = self.ho / h, self.wo / w

        fin_intrinsics = os.path.join(self.rootdir, 'intrinsic/intrinsic_color.txt')
        with open(fin_intrinsics, 'r') as freader:
            self.K = torch.tensor([float(i) for i in freader.read().replace('\n', ' ').split()]).view(4, 4)
        self.K[0, :] *= sw
        self.K[1, :] *= sh
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        idx = idx % len(self.image_paths)
        
        rgb_numpy = imutils.png2np( os.path.join( self.rootdir, 'color', '%d.jpg' % idx) )
        rgb = F.interpolate( torch.from_numpy(rgb_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                             (self.ho, self.wo),
                             mode="bilinear",
                             align_corners=True).squeeze(0)
        depth_scaling = cv2.imread( os.path.join( self.rootdir, 'depth', '%d.png' % idx), cv2.IMREAD_UNCHANGED ) / 1000.
        depth_scaling = F.interpolate(torch.tensor(depth_scaling).unsqueeze(0).unsqueeze(0), (self.ho, self.wo), mode='nearest').squeeze(0).float()

        # Estimate a monocular depth map
        inv_depth_mono = self.monocular_depth( rgb_numpy ) 
        inv_depth_mono = F.interpolate( torch.from_numpy(inv_depth_mono).unsqueeze(0).unsqueeze(0).float(),
                                        (self.ho, self.wo),
                                        mode='nearest').squeeze(0)
        #
        # Scale monocular depth map
        mask_scaling = depth_scaling > 1e-5
        inv_depth_scaling = 1 / depth_scaling
        inv_depth_scaling[~mask_scaling] = 0
        inv_depth_scaled, _, _ = utils.scale_depth( inv_depth_mono,
                                                    inv_depth_scaling,
                                                    mask_scaling )
        depth_scaled = (1 / inv_depth_scaled) 
        depth_cleaned = utils.clean_depth_edges(depth_scaled.squeeze(0)).unsqueeze(0)
        
        # Load the world-to-camera pose as a 4x4 matrix
        pose_file = os.path.join(self.rootdir, 'pose', '%d.txt' % idx)
        with open(pose_file, 'r') as freader:
            pose_c2w = np.array([float(i) for i in freader.read().replace('\n', ' ').split()]).reshape(4, 4)
        pose_w2c = np.linalg.inv(pose_c2w)

        # We use OpenGL's coordinate system: +ve x is to the right, +ve y is up, +ve z is out of the screen
        # ScanNet poses are in OpenCV's system so we perform a change of bases here.
        M = np.eye(4)
        M[:, 1:3] *= -1
        pose_w2c = torch.from_numpy(np.matmul(M, pose_w2c))
                
        return { "rgb": rgb,
                 "depth": depth_cleaned,
                 "pose_w2c": pose_w2c,
                 "K": self.K }
    

