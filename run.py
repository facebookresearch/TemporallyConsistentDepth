# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import numpy as np
import argparse
from datasets.scannet import ScanNetDataset
from datasets.colmap import ColmapDataset
from datasets.mpisintel import MPISintelDataset
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from pcupdate import PCUpdate
from utils import imutils
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", help="Base directory of ScanNet scene.")
    parser.add_argument("-o", "--outdir", default="output", help="Directory for saving output depth in.")
    parser.add_argument("-s", "--scene", default="alley_2", help="The name of the MPI Sintel scene to run the method on.")
    parser.add_argument("--save_numpy", action="store_true", help="Save the processed depthmaps as Numpy files.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--demo', action='store_true', help="Run the method on the provided demo data.")
    group.add_argument('--scannet', action='store_true', help="Run the method on a ScanNet scene.")
    group.add_argument('--colmap', action='store_true', help="Run the method on COLMAP data.")
    group.add_argument('--mpisintel', action='store_true', help="Run the method on a scene from the MPI-Sintel dataset.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    if (torch.cuda.is_available()):
        device = "cuda:0"
    else:
        print("No CUDA device found; Exiting.")
        exit()

    h, w = 484, 648
    if args.demo:
        dataset = ScanNetDataset(os.path.join(os.getcwd(), 'test_data'), h, w)
    elif args.scannet:
        dataset = ScanNetDataset(args.indir, h, w)
    elif args.colmap:
        dataset = ColmapDataset(args.indir, h, w)
    elif args.mpisintel:
        h, w = 436, 1024
        dataset = MPISintelDataset(args.indir, args.scene, h, w)
    else:
        print("Error: Set one of dataset flags '--scannet', '--colmap', '--mpisintel', or '--demo'. For custom datasets please see the README.")
        exit()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )
    pcupdater = PCUpdate(h, w, device)

    output_frames = []
    
    for batch, sample in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        
        pcupdater.update( sample['rgb'].to(device),
                          sample['depth'].to(device),
                          sample['pose_w2c'].to(device),
                          sample['K'].to(device) )

        depth_out = pcupdater.fused_depth.cpu()
        
        if batch == 0:
            mn, mx = torch.quantile(depth_out[depth_out > 0], 0.05), torch.quantile(depth_out[depth_out > 0], 0.95)

        depth_comparison_rgb = imutils.np2png_d( [ sample['depth'].view(h, w).cpu().numpy(),
                                                   depth_out.view(h, w).cpu().numpy() ],
                                                 fname=None,
                                                 vmin=mn,
                                                 vmax=mx )

        output = np.concatenate( (sample['rgb'].squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                  depth_comparison_rgb), 1 )
        output_frames.append( (output * 255).astype(np.uint8) )
        imutils.np2png( [ output ], os.path.join( args.outdir, '%.04d.png' % batch ))

        if args.save_numpy:
            np.save(os.path.join( args.outdir, '%.04d.npy' % batch), depth_out.numpy())

    video_clip = ImageSequenceClip(output_frames, fps=15)
    video_clip.write_videofile( os.path.join(args.outdir, 'output.mp4'), verbose=False, codec='mpeg4', logger=None, bitrate='2000k')
