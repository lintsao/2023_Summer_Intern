# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello, Yufeng Zheng.
# --------------------------------------------------------
import numpy as np
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from argparse import Namespace

from .densenet import DenseNetBlock, DenseNetTransitionUp, DenseNetDecoderLastLayers
from core import DefaultConfig
from encoder4editing_tmp.models.psp import pSp  # we use the pSp framework to load the e4e decoder.

config = DefaultConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):

    def __init__(self, num_all_embedding_features, pretrained_decoder):
        super(Decoder, self).__init__()

        self.decoder = pretrained_decoder

    def forward(self, embeddings):
        x = self.decoder(embeddings)
        return x