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

class ColmapDataset(Dataset):
    def __init__(self, rootdir, output_height=484, output_width=684):
        super(ColmapDataset, self).__init__()

        self.rootdir = rootdir
        self.ho, self.wo = output_height, output_width

        # Replace the following line with any desired monocular depth estimator.
        # Our method has been tested with DPT, and uses it by default. 
        self.monocular_depth = DPTWrapper(model_path='./DPT/weights/dpt_hybrid-midas-501f0c75.pt')

        poses_w2c, K, img_names = self.load_colmap()

        perm = np.argsort(img_names)
        self.poses_w2c = poses_w2c[perm, ...]
        self.K = K[perm, ...]
        self.img_names = [img_names[i] for i in perm]

        h, w = imutils.png2np( os.path.join( self.rootdir, 'images', self.img_names[0]) ).shape[:2]
        sh, sw = self.ho / h, self.wo / w

        self.K[:, 0, :] *= sw
        self.K[:, 1, :] *= sh

    def quaternion2mat(self, q):
        rot = np.zeros((3, 3))
        rot[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
        rot[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
        rot[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
        rot[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
        rot[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
        rot[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
        rot[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
        rot[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
        rot[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
        return rot
    
    def load_colmap(self):
        poses_w2c = []
        cameras = []
        img_names = []
        K = []

        fin = os.path.join( self.rootdir, 'sparse', 'cameras.txt' )
        with open(fin, 'r') as freader:

            for line in freader:
                tokens = line.split()
                if not tokens or tokens[0] == '#':
                    continue
                else:
                    # We want [id, fx, fy, cx, cy]
                    # The first set of cameras have only one focal length parameter
                    if tokens[1] == 'SIMPLE_PINHOLE' or tokens[1] == 'SIMPLE_RADIAL' or tokens[1] == 'RADIAL' or tokens[1] == 'SIMPLE_RADIAL_FISHEYE' or tokens[1] == 'RADIAL_FISHEYE':
                        cam = np.array([ float(tokens[0]), float(tokens[4]), float(tokens[4]), float(tokens[5]), float(tokens[6]) ]).reshape(1, -1)
                    else:
                        cam = np.array([ float(tokens[0]), float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7]) ]).reshape(1, -1)
                    cameras.append(cam)

        cameras = np.concatenate( cameras, 0 )
        cameras = cameras[ cameras[:, 0].argsort() ] # sort by IDs

        fin = os.path.join( self.rootdir, 'sparse', 'images.txt' )
        with open(fin, 'r') as freader:
            skipLine = False

            for line in freader:
                tokens = line.split()

                if skipLine:
                    skipLine = False
                    continue
                elif not tokens or tokens[0] == '#':
                    continue
                else:
                    qwxyz = np.array([float(i) for i in tokens[1:5]]) # rotation as a quaternion
                    txyz  = np.array([float(i) for i in tokens[5:8]]) # translation
                    cam_id = int(tokens[8]) # camera id is used to get intrinsics
                    
                    imfile = tokens[-1]
                    img_names.append(imfile)
                    pose_w2c = np.identity(4)
                    pose_w2c[0:3, 0:3] = self.quaternion2mat(qwxyz)
                    pose_w2c[0:3, -1]  = txyz
                    poses_w2c.append(pose_w2c)
                    
                    k = np.eye(4)
                    camera = cameras[cameras[:, 0] == cam_id, :].flatten()
                    k[0, 0] = camera[1] #cameras[cam_id, 1]
                    k[1, 1] = camera[2] #cameras[cam_id, 2]
                    k[0, 2] = camera[3] #cameras[cam_id, 3]
                    k[1, 2] = camera[4] #cameras[cam_id, 4]
                    K.append(k)
                    
                    skipLine = True
                    
        return [np.array(poses_w2c), np.array(K), img_names]


    # The code for reading COLMAP depth has been adapted from the COLMAP repository
    def read_colmap_depth(self, path):
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                    usecols=(0, 1, 2), dtype=int)
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        depth_map = np.transpose(array, (1, 0, 2)).squeeze()

        min_depth, max_depth = np.percentile(depth_map, [5, 95])
        depth_map[depth_map <= min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        return depth_map

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        idx = idx % len(self.img_names)

        rgb_numpy = imutils.png2np( os.path.join( self.rootdir, 'images', '%s' % self.img_names[idx]) )
        rgb = F.interpolate( torch.from_numpy(rgb_numpy).permute(2, 0, 1).unsqueeze(0).float(),
                             (self.ho, self.wo),
                             mode="bilinear",
                             align_corners=True).squeeze(0)

        depth_scaling = self.read_colmap_depth( os.path.join( self.rootdir,
                                                              'stereo/depth_maps',
                                                              '%s.geometric.bin' % self.img_names[idx]) )
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

        pose_w2c = self.poses_w2c[idx, ...]  # Load the world-to-camera pose as a 4x4 matrix
        K = self.K[idx, ...]

        # We use OpenGL's coordinate system: +ve x is to the right, +ve y is up, +ve z is out of the screen
        # Colmap poses are in OpenCV's system so we perform a change of bases here.
        M = np.eye(4)
        M[:, 1:3] *= -1
        pose_w2c = torch.from_numpy(np.matmul(M, pose_w2c))

        return { "rgb": rgb,
                 "depth": depth_cleaned,
                 "depth_gt": depth_scaling,
                 "pose_w2c": pose_w2c,
                 "K": K }
    
