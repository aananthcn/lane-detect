# Introduction
This project aims to compare and evaluate different Land Detection algorithms. This project uses datasets from https://universe.roboflow.com/techmrt/pavement-feature/model/2

<br>

# Getting Started

## Python Setup
`python3 -m venv venv`
`source venv/bin/activate`

<br>

## Python Installation
`pip install ultralytics opencv-python roboflow`

### For Linux
`sudo apt-get update`
`sudo apt-get install libxcb-xinerama0 libqt5gui5 libxkbcommon-x11-0 libxcb-cursor0`

<br>

# Train the model
 * `python util_train_lane.py`
 * `cp ./runs/segment/train/weights/best.pt ./my-ld-model.pt`


## Download the model, if you can't train
yolov11s: https://drive.google.com/file/d/1Rzfe2NsntfdErjt0HnNe4ebHg2a3tmgB/view?usp=sharing
Note: 
 * Click the link above and search for download icon / button to download
 * Then move the my-ld-model.pt to root folder of this repo folder cloned on your machine

<br>

# Sample Videos for Running
Indian road: https://drive.google.com/file/d/1NMzirobPER90bH2uDxGEJWzLn0H7r5xM/view?usp=sharing
US road: https://www.kaggle.com/datasets/dpamgautam/video-file-for-lane-detection-project?resource=download 

<br>

# Run the program
 * `python ld_yolo.py file ~/Downloads/test_video.mp4`
 * `python ld_yolo.py file road-travel-2025_1min.mp4`
