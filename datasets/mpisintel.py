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
from raftstereo_wrapper import RAFTStereoWrapper
import torch.nn.functional as F
import numpy as np
import glob
import cv2
import utils.imutils as imutils
import utils.utils as utils

class MPISintelDataset(Dataset):
    def __init__(self, rootdir, scene, output_height=436, output_width=1024):
        super(MPISintelDataset, self).__init__()

        self.rootdir = rootdir
        self.scene = scene
        self.ho, self.wo = output_height, output_width

        # Replace the following line with any desired stereo depth estimator.
        # Our method has been tested with RAFT-Stereo, and uses it by default. 
        self.stereo_depth = RAFTStereoWrapper(model_path='./RAFT-Stereo/models/raftstereo-realtime.pth')
            
        self.image_paths = glob.glob(self.rootdir + "/training/final_left/%s/*.png" % self.scene)
        
    # The code for cam_read() is taken from the MPI Sintel Depth dataset SDK
    def cam_read(self, filename):
        """ Read camera data, return (M,N) tuple.
        M is the intrinsic matrix, N is the extrinsic matrix, so that
        x = M*N*X,
        with x being a point in homogeneous image pixel coordinates, X being a
        point in homogeneous world coordinates.
        """
        TAG_FLOAT = 202021.25
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
        N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
        return M,N

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        idx = idx % len(self.image_paths)

        idx += 1 # MPI Sintel frame numbers start at 1
        
        # Left image
        left_numpy = imutils.png2np( os.path.join( self.rootdir, 'training', 'final_left', '%s/frame_%.04d.png' % (self.scene, idx)) )
        left = F.interpolate( torch.from_numpy(left_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                              (self.ho, self.wo),
                              mode="bilinear",
                              align_corners=True).squeeze(0)
        # Right image
        right_numpy = imutils.png2np( os.path.join( self.rootdir, 'training', 'final_right', '%s/frame_%.04d.png' % (self.scene, idx)) )
        right = F.interpolate( torch.from_numpy(right_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                              (self.ho, self.wo),
                              mode="bilinear",
                              align_corners=True).squeeze(0)

        # The disparity is computed here, after the images have been scaled to the output size
        disparity = torch.abs(self.stereo_depth( left, right ))
        
        # Load pose and camera intrinsics, scaling the latter if the output size of images is different than the original size
        K, pose_w2c = self.cam_read( os.path.join( self.rootdir, 'training', 'camdata_left', '%s/frame_%.04d.cam' % (self.scene, idx)) )
        pose_w2c = np.concatenate( (pose_w2c, np.array( [0, 0, 0, 1]).reshape(1, 4)), 0)
        
        h, w = left_numpy.shape[:2]
        sh, sw = self.ho / h, self.wo / w
        K[0, :] *= sw
        K[1, :] *= sh

        # Convert disparity to depth
        B = 0.1 # From the MPI Sintel dataset website, the baseline of the cameras = 10cm = 0.1m
        f = K[0, 0]
        depth = B * f / disparity.clamp(min=1e-10)

        depth_cleaned = utils.clean_depth_edges(depth.squeeze()).unsqueeze(0)

        # We use OpenGL's coordinate system: +ve x is to the right, +ve y is up, +ve z is out of the screen
        # MPI-Sinterl poses are in Blender system so we perform a change of bases here.
        M = np.eye(4)
        M[:, 1:3] *= -1
        pose_w2c = torch.from_numpy(np.matmul(M, pose_w2c))

        return { "rgb": left,
                 "depth": depth_cleaned,
                 "pose_w2c": pose_w2c,
                 "K": torch.from_numpy(K) }
    

