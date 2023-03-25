# Multi-Task Neural Network for Auonomous Driving

Rishav Kumar Paramhans

# Introduction

* The Multi-task Neural Network performs 2D Object Detection, Monocular 3D Object Detection, Semantic Segmentation and Monocular Depth Estimation simaltaneously for autonomous driving senarios. 
* The model has been training on Audi Autonomous Driving Dataset.
* The network outperforms the corresponding single-task neural network by susbtatial margin. 
* Metrics used for performance quantification: a) Monocular 3D Object Detection: Mean Average Precsion BEV
                                               b) 2D Object Detection : Mean Average Precision
                                               c) Semantic Segmentation : DICE Score amd IOU Score
                                               d) Depth Estimation: Threshold ( 0.5, 0.75 of Standard Deviations)

If you utilize this work, please give a star to this repository.
A detailed report of the project will be uploaded under the assets folder soon.

# Setup
* # Cuda and Python
In this Project I utilized Pytorch with Python 3.10, Cuda 11.6 and few other python libraries. However, feel free to try alternative versions or model of installation.

* # Data
Download the full A2D2 dataset from the official Audi Dataset website. Training and Test Split of the data can be found in this repository as .CSV files under the name 'train.csv' and 'test.csv'.

* # Training
Use the command: python train.py
The training progress can be monitored using Tensorboard. Writing the logs on the Tensorboard can be handled using hthe hypermeter WRITER in the training scripts.

* # Contact
For questions reagrding the project, feel free to post here or directly contact the author at rishavkrparamhans@gmail.com
