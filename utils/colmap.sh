#!/bin/bash

dir_in=$1
dir_out=$2

colmap feature_extractor --database_path "$dir_out/database.db" --image_path "$dir_in" \
       --SiftExtraction.use_gpu=1 \
       --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true \
       --ImageReader.single_camera=1 --ImageReader.camera_model='SIMPLE_PINHOLE' 
colmap exhaustive_matcher --database_path "$dir_out/database.db" # --SiftMatching.guided_matching=true
# colmap sequential_matcher --database_path "$dir/database.db" --SiftMatching.guided_matching=true
mkdir "$dir_out/sparse"
colmap mapper --database_path "$dir_out/database.db" --image_path "$dir_in" --output_path "$dir_out/sparse"
colmap model_converter --input_path "$dir_out/sparse/0" --output_path "$dir_out/sparse/0" --output_type=TXT

colmap image_undistorter --image_path "$dir_in" --input_path "$dir_out/sparse/0" --output_path "$dir_out/"
colmap patch_match_stereo --workspace_path "$dir_out/" --PatchMatchStereo.gpu_index 0

colmap model_converter --input_path "$dir_out/sparse" --output_path "$dir_out/sparse" --output_type=TXT
