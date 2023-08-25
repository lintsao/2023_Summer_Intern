#!/bin/bash

mkdir -p pretrained_models

# Replace with your file's ID and desired output filename
baseline_estimator_resnet_file_id="1P4PnRMDhb37NXnezYosiwqCQrEguD2kd"

baseline_estimator_vgg_file_id="1amWI-1mrVIRLgUntnvBwuAj3Nn9ktiq9"

e4e_file_id="1cUv_reLE6k3604or78EranS7XzuVMWeO"

r50_file_id="1UyqKMdCdVNfeXnPT7rP-QqLaauudCSGJ"

# Use curl to download the file
gdown --id "$baseline_estimator_resnet_file_id" -O "pretrained_models/baseline_estimator_resnet.tar"

gdown --id "$baseline_estimator_vgg_file_id" -O "pretrained_models/baseline_estimator_vgg.tar"

gdown --id "$e4e_file_id" -O "pretrained_models/e4e_ffhq_encode.pt"

gdown --id "$r50_file_id" -O "pretrained_models/r50_backbone.pth"