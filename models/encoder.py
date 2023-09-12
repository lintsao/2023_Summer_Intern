# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello, Yufeng Zheng.
# --------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from argparse import Namespace
import sys

from .densenet import (
    DenseNetInitialLayers,
    DenseNetBlock,
    DenseNetTransitionDown,
)
from core import DefaultConfig
from encoder4editing_tmp.models.psp import pSp  # we use the pSp framework to load the e4e encoder.

config = DefaultConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, pretrained_encoder):
        super(Encoder, self).__init__()

        self.encoder = pretrained_encoder

    def forward(self, image):
        x = self.encoder(image) # image: [batch, 3, 256, 256]
        batch_size = x.shape[0]
        x = x.contiguous().view(batch_size, -1) # image: [batch, 512 * 18]

        return x