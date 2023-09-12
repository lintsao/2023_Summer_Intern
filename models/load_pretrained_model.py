import torch
import torchvision.transforms as transforms
from argparse import Namespace

from encoder4editing_tmp.models.psp import pSp  # we use the pSp framework to load the e4e encoder.
from insightfacemodule_tmp.recognition.arcface_torch.backbones import get_model

import cv2
import numpy as np



def load_pretrained_model(load_e4e_pretrained = True):
    EXPERIMENT_DATA_ARGS = {
        "e4e": "pretrained_models/e4e_ffhq_encode.pt",
        "r50": "pretrained_models/r50_backbone.pth"
    }

    # Setup required image transformations
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS
    EXPERIMENT_ARGS['e4e_transform'] = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    EXPERIMENT_ARGS['r50_transform'] = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((112, 112))])

    ckpt = torch.load(EXPERIMENT_ARGS['e4e'], map_location='cuda')
    opts = ckpt['opts']
    if load_e4e_pretrained:
        opts['checkpoint_path'] = EXPERIMENT_ARGS['e4e']
    else:
        opts['checkpoint_path'] = None

    opts = Namespace(**opts)
    e4e_net = pSp(opts)
    e4e_net.eval()
    e4e_net.cuda()

    r50_net = get_model('r50', fp16=True)
    r50_net.load_state_dict(torch.load(EXPERIMENT_DATA_ARGS['r50']))
    r50_net.eval()
    r50_net.cuda()

    return e4e_net, r50_net, EXPERIMENT_ARGS['e4e_transform'], EXPERIMENT_ARGS['r50_transform']