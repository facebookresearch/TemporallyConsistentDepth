# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """ 
    This class provides the template for a custom dataset.
    The __getitem__() function is expected to return the unprocessed depth map, the RGB image, and the camera pose and intrinsics.
    Important points to note are:
    - The data loader should return a *scaled* depth map.
    - The camera poses should be in a right-handed coordinate system similar to OpenGL. That is, the X axis pointing to the right, 
    the Y axis pointing upwards, and the Z axis coming out of the screen.
    """
    def __init__(self, rootdir, output_height=484, output_width=684):
        super(CustomDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return { "rgb": None,
                 "depth": None,
                 "pose_w2c": None,
                 "K": None}
    

