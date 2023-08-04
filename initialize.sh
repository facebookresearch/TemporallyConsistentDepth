#!/bin/bash

# Download backbone monocular and stereo depth estimation models
# We use DPT and RAFT-Stereo by default but the code can work with any depth estimation model

git submodule update --init --recursive

# Download backbone weights
mkdir ./RAFT-Stereo/models -p
wget https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip -O ./RAFT-Stereo/models/models.zip -q --show-progress
unzip ./RAFT-Stereo/models/models.zip -d ./RAFT-Stereo/models/
rm ./RAFT-Stereo/models/models.zip -f

mkdir ./DPT/weights -p
wget 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt' -O ./DPT/weights/dpt_hybrid-midas-501f0c75.pt -q --show-progress
