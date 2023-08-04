# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import torch
import numpy as np
import cv2
from argparse import Namespace
sys.path.append( os.path.join(os.getcwd(), 'RAFT-Stereo'))
sys.path.append( os.path.join(os.getcwd(), 'RAFT-Stereo', 'core'))
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from collections import OrderedDict

# Source code adapted from https://github.com/princeton-vl/RAFT-Stereo
class RAFTStereoWrapper:

    def __init__(self, model_path):
        super(RAFTStereoWrapper, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_iters = 7
        
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Initialize RAFT-Stereo model parameters
        args = Namespace(
            mixed_precision=True,
            hidden_dims=[128]*3,
            corr_implementation="reg_cuda",
            shared_backbone=True,
            context_norm="batch",
            corr_levels=4,
            slow_fast_gru=True,
            corr_radius=4,
            n_downsample=3,
            n_gru_layers=2
        )

        # Create model
        model = RAFTStereo(args)
        
        # RAFT-Stereo checkpoints are saved as nn.DataParallel.
        # So we create a new OrderedDict that does not contain the 'module.' prefix
        # Code from:
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        self.model = model.to(self.device)
        self.model.eval()

        
    def __call__( self, left, right ):
        """ 
        Estimate stereo depth for the given left/rigth RGB input image pair.
        The input image values should be in the range [0, 1].
        """
        with torch.no_grad():
            left = (left * 255.).to(self.device).unsqueeze(0)
            right = (right * 255.).to(self.device).unsqueeze(0)

            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)

            _, prediction = self.model(left, right, iters=self.valid_iters, test_mode=True)
            prediction = padder.unpad(prediction)
            
        return prediction 


