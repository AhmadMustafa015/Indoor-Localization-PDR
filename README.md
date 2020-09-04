# Activity Recognition & Indoor Localization
This repository contains a dataset for activity recognition, a Jupyter notebook for a full data pre-processing and filtering on the dataset, multiple deep neural networks with LSTM layer for human activity prediction, a hyperparameter tuning algorithm to optimize the deep learning model, an android app for activity recognition that equipped with the pre-trained model as a TensorFlow lite file (.tflite) and an android app for step counting and floor estimation. Moreover, a full indoor navigation solution that built and tested on the android environment.

Folders are divided as follows:

* Documentation: this folder contains a report which describes the technical details of the algorithm and filters that are used in the indoor localization app.
* Human Activity Recognition: this folder contains the python scripts for the activity recognition training model, the dataset, and the pre-processing Jupyter notebook. Also, it contains presentation files that describe the models used and the results of the training.
* IndoorLocalization: this folder contains the codes for the indoor navigation android app.
* MVP: this folder has multiple videos to describe the app and how to use it (note the app version in the videos is an old one, the newer version has many updates).
* References: this folder contains the reference papers used as a foundation for this work.
* stepDetectionAndStepEstimation: the folder that contains the android app for the step counting, stride length estimation, and floor detection algorithm.

## Part 1: Activity Recognition

### Introduction
In Activity recognition part we develop an deep learning model using tensorflow and an android app to test the trained model. The datasets used in the work is from UCI machine learning reporitory link: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones and the data collected by Serhat. The detected gystures are (walking, going upstairs, going downstairs, standing, running, elevator, falling, car in or out).The trained model was converted to tensorflow lite file (.tflite) to use in the smartphone. 

There are three training phases was done using the final model at each stage we try defferent datasets and features to enhance the confusion matrix and therefor enhance the overall system.

#### * First training phase: 
Datasets: using Serhat collected dataset + UCI dataset. 

The features: features that used in this part are (acceleration X, acceleration Y, acceleration Z, gyroscope X, gyroscope Y, gyroscope Z, body acceleration X, body acceleration Y, body acceleration Z, RMS acceleration, RMS gyroscope, RMS body acceleration).

Results: Training accuracy: 97.57% >>> Validation accuracy: 96.59% >>> Test accuracy: 96.81%
![Picture1](https://user-images.githubusercontent.com/43111249/92290014-4a6c7a00-ef1b-11ea-90d7-e8ca61b028f7.png)

There are three main training phases









## Part 2: Indoor Localization Based on Pedestrian Dead Reckoning (PDR)

