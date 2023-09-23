# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello, Yufeng Zheng.
# --------------------------------------------------------
import numpy as np
from collections import OrderedDict
import gc
import json
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import logging
import losses
from tqdm import tqdm
import sys
import argparse
from dataset import HDFDataset
from utils import save_images, worker_init_fn, send_data_dict_to_gpu, recover_images, def_test_list, RunningStatistics,\
    adjust_learning_rate, script_init_common, get_example_images, save_model, load_model
from core import DefaultConfig
from models.st_ed_adv_pretrained_label import STED
import torchvision.transforms as transforms
from PIL import Image
import dlib
from encoder4editing_tmp.utils.alignment import align_face
import time
import math

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

config = DefaultConfig()
script_init_common()

if not os.path.exists(config.redirect_path):
    os.makedirs(config.redirect_path)
logging.info(f"Current saved path: {config.redirect_path}")

logging.info(f"Current input path: {config.input_path}")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
logging.info(f"Current device: {device}")

import warnings
warnings.filterwarnings('ignore')

# Create redirection network
network = STED(device).to(device)
# Load weights if available
from checkpoints_manager import CheckpointsManager

if config.checkpoint_path:
    model_path = os.path.join(config.checkpoint_path)
    print("Load model from", model_path)
    load_model(network, model_path)
    logging.info("Loaded checkpoints done")

# Transfer on the GPU before constructing and optimizer
if torch.cuda.device_count() > 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    # network.redirtrans_p = nn.DataParallel(network.redirtrans_p)
    # network.redirtrans_dp = nn.DataParallel(network.redirtrans_dp)
    # network.fusion = nn.DataParallel(network.fusion)
    # network.discriminator = nn.DataParallel(network.discriminator)
    # network.GazeHeadNet_eval = nn.DataParallel(network.GazeHeadNet_eval)
    # network.GazeHeadNet_train = nn.DataParallel(network.GazeHeadNet_train)
    # network.lpips = nn.DataParallel(network.lpips)
    # network.pretrained_arcface = nn.DataParallel(network.pretrained_arcface)
    # network.e4e_net = nn.DataParallel(network.e4e_net)

# Inference.
# process data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

original_image = Image.open(config.input_path)
input_image = original_image.convert("RGB")

def run_alignment(image_path):
    predictor = dlib.shape_predictor("./pretrained_models/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    if aligned_image:
        logging.info("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image
    else:
        return None
    
def degree_to_rad(degrees):
    radians = degrees * (math.pi / 180)
    return radians

def rad_to_degree(radians):
    degrees = radians * (180 / math.pi)
    return degrees

if not config.preprocess:
    input_image = run_alignment(config.input_path)

transformed_image = transform(input_image)
all_data = OrderedDict()
all_data['image_a'] = transformed_image.unsqueeze(0)
# all_data['head_pitch'] = config.head_pitch
# all_data['head_yaw'] = config.head_yaw
# all_data['gaze_pitch'] = config.head_pitch
# all_data['gaze_yaw'] = config.gaze_yaw

degree_list = [-10.0, 0.0, 10.0]

for all_data['head_pitch'] in degree_list:
    for all_data['head_yaw'] in degree_list:
        for all_data['gaze_pitch'] in degree_list: 
            for all_data['gaze_yaw'] in degree_list:
                all_data['head_b_r'] = torch.cat([torch.tensor(degree_to_rad(all_data['head_pitch'])).unsqueeze(0), torch.tensor(degree_to_rad(all_data['head_yaw'])).unsqueeze(0)], dim=0).unsqueeze(0)
                all_data['gaze_b_r'] = torch.cat([torch.tensor(degree_to_rad(all_data['gaze_pitch'])).unsqueeze(0), torch.tensor(degree_to_rad(all_data['gaze_yaw'])).unsqueeze(0)], dim=0).unsqueeze(0)

                # - 90 degree ~ + 90 degree: - 1.57 rad ~ + 1.57 rad
                with torch.no_grad():
                    test_losses = RunningStatistics()
                    network.eval()
                    send_data_dict_to_gpu(all_data, device)
                    curr_time = time.time()
                    output_dict, loss_dict, label_dict = network.redirect(all_data)
                    print ("Inference time: %f " % (time.time() - curr_time))

                    for key, value in label_dict.items():
                        logging.info(f"{key} in degree: pitch {value[0].detach().cpu().numpy()}, yaw {value[1].detach().cpu().numpy()}")

                    for key, value in loss_dict.items():
                        test_losses.add(key, value.detach().cpu().numpy())
                    test_loss_means = test_losses.means()
                    logging.info('Test Losses current model for : %s' %((['%s: %.6f' % v for v in test_loss_means.items()])))

                    save_images(all_data['image_a'][0], os.path.join(config.redirect_path, 'input_image.png'), fromTransformTensor=True)
                    save_images(output_dict['image_a_inv'][0], os.path.join(config.redirect_path, 'input_inv_image.png'), fromTransformTensor=True)
                    save_images(output_dict["output_1024"][0], os.path.join(config.redirect_path, 
                                                                            f"hp_{all_data['head_pitch']}_hy_{all_data['head_yaw']}_gp_{all_data['gaze_pitch']}_gy_{all_data['gaze_yaw']}_1024.png"), fromTransformTensor=True)
                    save_images(output_dict["output_256"][0], os.path.join(config.redirect_path, 
                                                                           f"hp_{all_data['head_pitch']}_hy_{all_data['head_yaw']}_gp_{all_data['gaze_pitch']}_gy_{all_data['gaze_yaw']}_256.png"), fromTransformTensor=True)

                print("Finish Inference:", f"hp_{all_data['head_pitch']}_hy_{all_data['head_yaw']}_gp_{all_data['gaze_pitch']}_gy_{all_data['gaze_yaw']}")