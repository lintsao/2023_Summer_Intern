# Self-Learning Transformations for Improving Gaze and Head Redirection

This repository is the official implementation of [Self-Learning Transformations for Improving Gaze and Head Redirection](https://arxiv.org/abs/2010.12307), NeurIPS 2020. 

* Authors: [Yufeng Zheng](https://ait.ethz.ch/people/zhengyuf/), [Seonwook Park](https://ait.ethz.ch/people/spark/), [Xucong Zhang](https://ait.ethz.ch/people/zhang/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta) and [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)
* Project page: [https://ait.ethz.ch/projects/2020/STED-gaze/](https://ait.ethz.ch/projects/2020/STED-gaze/)

## Requirements
We tested our model with Python 3.8.3 and Ubuntu 16.04. The results in our paper are obtained with torch 1.3.1, but we have tested this code-base also with torch 1.7.0 and achieved similar performance. We provide the pre-trained model obtained with the updated version.

First install torch 1.7.0, and torchvision 0.8.1 following the guidance from [here](https://pytorch.org/get-started/locally/), and then install other packages:

```
pip install -r requirements.txt
```
 
To pre-process datasets, please follow the instructions of [this repository](https://github.com/swook/faze_preprocess). 
Note that we use full-face images with size128x128.

You can download the preprocessed data of GazeCapture and MPIIGaze
- [GazeCapture](https://drive.google.com/file/d/1hYgs770CcwLLD9Z7H-cjV8QzvMGVE9CS/view?usp=sharing)
- [MPIIGaze](https://drive.google.com/file/d/120zI6mZPr28SEm5jdHNuBXeHkEvC8Qu2/view?usp=sharing)

## File / Folder Description
```
Root
├──config
│   ├── eval.json: Config for evaluation
│   └── semi-supervise.json: Config for the semi-supervision gaze and head evaluation task
│   └── ST-ED.json: Config for the training task
├── configs
|
├── core
|
├── encoder4editing: Modules for the work, e4e
|
├── insightfacemodule: Modules for the work, face recognition
|
├── models
│   ├── decoder.py: For image decoder (Use pretrained)
│   └── densenet.py: For original ReDirTrans (Not used in our case)
│   └── discriminator.py: For original ReDirTrans
│   └── encoder.py: For image encoder (Use pretrained)
│   └── gazeheadnet.py: For VGG gaze estimator
│   └── gazeheadResnet.py: For Resnet gaze estimator
│   └── load_pretrained_model.py: Load pretrained model into the training pipeline.
│   └── redirtrans.py: Implement the ReDiiTrans module.
│   └── st_ed.py: Overall model.
|
├── ours: Virtual environment modules.
|
├── output: Virtual environment modules.
|
├── pretrained_models
│   ├── baseline_estimator_resnet.tar: resnet based gaze / head estimator
│   ├── baseline_estimator_vgg.tar: vgg based baseline_estimator_resnet.tar
│   ├── e4e_ffhq_encode.pt: e4e pretrained model
│   └── r50_backbone.pth: For or50 recognition model
│
├── __init__.py
│
├── .gitignore
│
├── checkpoints_manager.py
│
├── dataset_explore.ipynb: Check the dataset information.
│
├── dataset_augmented.py
│
├── dataset.py: Dataset class object.
│
├── gazecapture_split.json: Split the training / val / testing data.
│
├── losses.py: Function for loss.
|
├── main.py
|
├── README.md
|
├── requirements.txt
|
├── src.tar
|
├── train_facenet.py
|
├── train_st_ed.ipynb: Overall training pipeline in ipynb file.
|
├── train_st_ed.py: Overall training pipeline in py file.
|
├── utils.py: Useful functions.
```

## Usage
All available configuration parameters are defined in core/config_default.py.
In order to override the default values, one can do:

1. Pass the parameter via a command-line parameter. Please replace all `_` characters with `-`.
2. Create a JSON file such as `config/st-ed.json`.

The order of application are:

1. Default parameters
2. JSON-provided parameters
3. CLI-provided parameters

### Training

To train the gaze redirection model in the paper, run this command:

```
python train_st_ed.py config/ST-ED.json
```

You can check Tensorboard for training images, losses and evaluation metrics. Generated images from testsets are store in the model folder.

To train in a semi-supervised setting and generate augmented dataset, run this command (set ```num_labeled_samples```to a desired value):

```
python train_st_ed.py config/semi-supervise.json
```

Note that for semi-supervised training, we also train the estimator with only labeled images. We provide the script for training gaze and head pose estimators: ```train_facenet.py```, so that you can train baseline and augmented estimators and evaluate the data augmentation performance of our method. 

Training of redirector will take 1-2 days on a single GPU.

### Evaluation

To evaluate pretrained full model, run:

```
python train_st_ed.py config/eval.json
```

Quantitative evaluation of all test datasets will take a few hours. If you want to speed up the process, try to disable the calculation of disentanglement metrics, or evaluate on partial dataset (this is what we do during training!)
## Pre-trained Models

You can download pretrained models here:

- [Our fully-supervised gaze redirector model](https://drive.google.com/file/d/1PGb1GKy31WE692rvk_iBYQdeO_OK9BRi/view?usp=sharing)
- [VGG gaze estimator](https://drive.google.com/file/d/1amWI-1mrVIRLgUntnvBwuAj3Nn9ktiq9/view?usp=sharing)
- [ResNet gaze estimator for evaluation](https://drive.google.com/file/d/1P4PnRMDhb37NXnezYosiwqCQrEguD2kd/view?usp=sharing)
- [e4e encoder and decoder](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)
- [insightface r50 recognition model](https://drive.google.com/file/d/1UyqKMdCdVNfeXnPT7rP-QqLaauudCSGJ/view?usp=sharing)

## License
This code base is dual-licensed under GPL or MIT licenses, with exceptions for files with NVIDIA Source Code License headers which are under Nvidia license.

