
# Temporally Consistent Online Depth Estimation Using Point-Based Fusion
 [Numair Khan](https://nkhan2.github.io)<sup>1</sup>,
 Eric Penner<sup>1</sup>,
 Douglas Lanman<sup>1</sup>,
 [Lei Xiao](https://leixiao-ubc.github.io/)<sup>1</sup><br>
 <sup>1</sup>Reality Labs Research<br>
 CVPR 2023
       
### [Paper](https://arxiv.org/abs/2304.07435) | [Supplemental](https://research.facebook.com/publications/temporally-consistent-online-depth-estimation-using-point-based-fusion/) | [Video](https://www.youtube.com/watch?v=xlD6zywC0dI) 

## Running the Code
* [Setup](#setup)
* [Demo](#demo)
* [Running on ScanNet Data](#running-on-scannet)
* [Running on COLMAP Data](#running-on-colmap-data)
* [Running on MPI-Sintel](#running-on-mpi-sintel)
* [Running on Custom Data](#running-on-custom-data)
* [Troubleshooting](#troubleshooting)

### Setup
The code has been tested in the following setup
* Linux (Ubuntu 20.04.04/Fedora 34)
* Python 3.7
* PyTorch 1.10.2
* CUDA 11.3

We recommend running the code in a virtual environment such as Conda. After cloning the repo, run the following commands from the base directory:
```
$ conda env create -f environment.yml
$ conda activate tcod
```

Then, run the following script to download model checkpoints and set up the depth estimation backbones. We use [DPT](https://github.com/isl-org/DPT) and
[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) as the monocular and stereo depth estimation backbones respectively.

```
$ sh ./initialize.sh
```

### Demo
A small subset of ScanNet data is included in the `test_data` folder for demoing and testing purposes. To execute the code on it, run
```
$ python ./run.py --demo
```
By default, this script will generate a visual comparison of the results (RGB | Input Depth | Our Result) in a folder called `output`. You can use the `--save_numpy` flag to save the actual floating point depth values to separate files. A different output location can be specified by setting the `--outdir` argument.

### Running on ScanNet
To run the method with monocular depth estimation on a ScanNet scene use:

`$ python ./run.py --scannet --indir=/ScanNet/scans/sceneXXXX_XX --outdir=/PATH_TO_OUTPUT_DIR`

We assume the data is extracted into the standard ScanNet directories (If this is not the case modify the paths used in `datasets/scannet.py`). 

### Running on COLMAP Data 
To run the method with monocular depth estimation on COLMAP data use:

`$ python ./run.py --colmap --indir=/PATH_TO_COLMAP_DIR/ --outdir=/PATH_TO_OUTPUT_DIR`

Again, we assume all data exists in standard COLMAP directory format. To ensure this, we recommend using the provided COLMAP execution script:

`$ sh ./utils/colmap.sh PATH_TO_VIDEO_FRAMES  PATH_TO_COLMAP_DIR`

Depending on the image sequence, the parameters used at different stages of COLMAP may need to be adjusted to generate a good reconstruction. 

### Running on MPI-Sintel
To run the method with stereo depth estimation on the MPI-Sintel dataset use:

`$ python ./run.py --mpisintel --scene SCENE_NAME --indir=/PATH_TO_MPI_SINTEL_BASE_DIR/ --outdir=/PATH_TO_OUTPUT_DIR`

Where `SCENE_NAME` is one of the 23 scenes in the dataset derived from the respective folder names (`alley_1`, `alley_2`, etc). We assume the user has downloaded the depth/camera motion and the stereo/disparity training data from the dataset and extracted them into a single folder at `PATH_TO_MPI_SINTEL_BASE_DIR`. 

### Running on Custom Data

To test the method with custom data (stereo or monocular) you will need to implement a data loader based on the template provided in `datasets/custom.py` and add it in `run.py`. Please refer to the loaders provided for the above-mentioned datasets as examples. In brief, the data loader is expected to return an unprocessed depth map, the RGB image, and the camera pose and intrinsics.

Important points to note are
1. The data loader should return a **scaled** depth map. 
2. The camera poses should be in a right-handed coordinate system similar to OpenGL. That is, the *X* axis pointing to the right, the *Y* axis pointing upwards, and the *Z* axis coming out of the screen.

### Troubleshooting
Some common causes of large errors in the output can be:
* Using the incorrect coordinate system for the camera poses. Our method assumes *Y* points upwards and *Z* points out of the screen. OpenCV, COLMAP and Blender, on the other hand, assume *Y* is down and *Z* is forward.  
* Using disparity instead of depth as the input. Most stereo algorithms will return a disparity map which should be converted to depth. Assuming the left and right cameras have similar intrinsics, this conversion can be done using `depth = f * B / disparity`, where `f` is the horizontal focal length and `B` is the stereo baseline.
* Not using scaled depth. Monocular depth estimation methods are scale-ambiguous and require a scaling factor (coming from, for example, a ToF or LiDAR sensor, or SfM) to generate correct results. 
* The method has been designed and tested on video data with relatively small camera movement between consecutive frames. It may not generalize to arbitrary multi-view datasets with large changes in camera pose.

## Errata
Will be added as required.

## Citation
If you find our work useful for your research, please cite the following paper:

```
@article{khan2023tcod,
  title={Temporally Consistent Online Depth Estimation Using Point-Based Fusion},
  author={Numair Khan, Eric Penner, Douglas Lanman, Lei Xiao},
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```

## License
Our source code is CC-BY-NC licensed, as found in the LICENSE file.