# Getting Started
 * clone this repo as lane-detect
 * cd lane-detect
 * git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git ufld
 * mkdir -p ./ufld/model/weights



*Note*: the UFLDv2 algorithm is based on paper "**Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification**" Link: https://arxiv.org/pdf/2206.07389


## Create virtual python environment
 * python -m venv venv
 * source venv/bin/activate


## Installation
 * pip install opencv-python torch torchvision opencv-python numpy scipy
 * pip install gdown
 * pip install addict tensorboard opencv-python tqdm albumentations
 * pip install pathspec scipy pyyaml


## Download Tusimple ResNet18 model (good starter, ~96% acc)
 * gdown 1Clnj9-dLz81S3wXiYtlkc4HVusCb978t -O ufld/model/weights/tusimple_res18.pth


## Video source
 * US Road: https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project?resource=download
 * Indian Road: https://drive.google.com/file/d/1NMzirobPER90bH2uDxGEJWzLn0H7r5xM/view?usp=sharing

### Run the example
 * python ld_ufdld_hadoc.py file road-travel-2025_1min.mp4