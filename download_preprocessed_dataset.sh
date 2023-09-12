#!/bin/bash

mkdir -p dataset

# Replace with your file's ID and desired output filename
GazeCapture_128_file_id="1hYgs770CcwLLD9Z7H-cjV8QzvMGVE9CS"
# GazeCapture_output_file="dataset/GazeCapture_128.h5"

MPIIGaze_128_file_id="120zI6mZPr28SEm5jdHNuBXeHkEvC8Qu2"
# MPIIGaze_output_file="dataset/MPIIGaze_128.h5"

# # URL to download from Google Drive
# GazeCapture_url="https://drive.google.com/uc?export=download&id=$file_id"
# MPIIGaze_url="https://drive.google.com/uc?export=download&id=$file_id"

# Use curl to download the file

gdown --id "$GazeCapture_128_file_id" -O "dataset/GazeCapture_128.h5"

gdown --id "$MPIIGaze_128_file_id" -O "dataset/MPIIGaze_128.h5"

# Replace with your file's ID and desired output filename
GazeCapture_256_file_id="1JWdCiFxVkNytAPetsm4alV2OQAxKbiHO"
# GazeCapture_output_file="dataset/GazeCapture_128.h5"

MPIIGaze_256_file_id="1YVjsqFkv65orlFRG_Eb-ZkDnkcXEs4AW"
# MPIIGaze_output_file="dataset/MPIIGaze_128.h5"

# Use curl to download the file

gdown --id "$GazeCapture_256_file_id" -O "dataset/GazeCapture_2561.h5"

gdown --id "$MPIIGaze_256_file_id" -O "dataset/MPIIGaze_2561.h5"