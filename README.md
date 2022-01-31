# YOLOv3: Vehicle detection and classification
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

A simple demonstration of real-time vehicle detection and classification using YOLO3 using cv2 DNN backend. Vehicles are broadly classified into trucks, bus, car, motobike.

<div style="text-align:center">
  <figure>
    <img src=./asset/output.gif>
  <figure>
</div>

## Table of Contents
<!-- MarkdownTOC autolink="true" bracket="round" autoanchor="false" lowercase="only_ascii" uri_encoding="true" levels="1,2,3,4" -->
1. [Getting Started](#getting-started)
    - [Setup](#setup)
    - [Configuration](#configuration)
2. [Usage](#usage)
3. [Acknowledgements](#acknowledgements)
<!-- /MarkdownTOC -->

## Getting Started
Follow the instructions down below to get started on the pre-installation guidelines and other project related configurations.
### Setup

Setup and install necessary dependencies,
```
pip install -U pip
pip install -r requirements.txt
```

### Configuration
Download yolo3 model configuration (yolov3.cfg), weights (yolov3.weights), labels (coco.names) from [YOLO](https://pjreddie.com/darknet/yolo/).
Add the above paths along with image/video path in the config.toml file (an example configuration file has been added in the repository).
Default value for threshold and confidence is 0.5 and 0.3 respectively.


## Usage
Takes an image/video as input, resizes the image/frame before passing through yolo detector. Detections are then classified into cars, trucks, buses, motorbikes among the classes in label file. Output image/video is saved in the project folder.
To make predictions run,
```
python make_predictions.py
```
<div style="text-align:center">
  <figure>
    <img src=./asset/output.jpg>
  <figure>
</div>

## Acknowledgements

This project is inspired by [yolo-object-detection](https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python).

