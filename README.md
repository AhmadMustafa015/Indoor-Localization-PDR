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
In the Activity recognition part, we develop a deep learning model using TensorFlow and an android app to test the trained model. The datasets used in the work are from UCI machine learning repository link: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones and the data collected by Serhat. The detected gestures are (walking, going upstairs, going downstairs, standing, running, elevator, falling, car in or out). The trained model was converted to a tensorflow lite file (.tflite) to use in the smartphone. 

Three training phases were done using the final model at each stage we try different datasets and features to enhance the confusion matrix and therefore enhance the overall system.

#### First training phase: 
Datasets: using Serhat collected dataset + UCI dataset. 

The features: features that are used in this part are (acceleration X, acceleration Y, acceleration Z, gyroscope X, gyroscope Y, gyroscope Z, body acceleration X, body acceleration Y, body acceleration Z, RMS acceleration, RMS gyroscope, RMS body acceleration).

Results: Training accuracy: 97.57% >>> Validation accuracy: 96.59% >>> Test accuracy: 96.81%

![Picture1](https://user-images.githubusercontent.com/43111249/92290014-4a6c7a00-ef1b-11ea-90d7-e8ca61b028f7.png)

#### Second training phase: 
Datasets: using Serhat collected dataset only

The features: features that are used in this part are (acceleration X, acceleration Y, acceleration Z, gyroscope X, gyroscope Y, gyroscope Z, body acceleration X, body acceleration Y, body acceleration Z, RMS acceleration, RMS gyroscope, RMS body acceleration).

Results: Training accuracy: 88.52% >>> Validation accuracy: 86.89% >>> Test accuracy: 87.42%

#### Third training phase: 
Datasets: using Serhat collected dataset only

The features: features that used in this part are (acceleration X, acceleration Y, acceleration Z, gyroscope X, gyroscope Y, gyroscope Z, body acceleration X, body acceleration Y, body acceleration Z, RMS acceleration, RMS gyroscope, RMS body acceleration, magnetometer X, magnetometer Y, magnetometer Z).

Results: Training accuracy: 96.02% >>> Validation accuracy: 95.97% >>> Test accuracy: 98.11%

![Picture2](https://user-images.githubusercontent.com/43111249/92290029-55bfa580-ef1b-11ea-8314-441537567300.png)

As seen using the magnetometer helped in reducing the confusion between the Fallin the Elevator gestures.

#### Important note: unfortunately only Serhat collected data has magnetometer readings so in the final version we couldn't use the UCI dataset. Also, the Serhat collected data from foot only, so to have more accurate results in real-time applications, more data need to be collected from different people. Besides, the collected data need to be collected from different positions (for example waist, pocket, handheld).

## Part 2: Indoor Localization Based on Pedestrian Dead Reckoning (PDR)

### Inroduction
The indoor localization app is based on Pedestrian Dead Reckoning (PDR), which uses a step detection and stride length estimation algorithm to count the number of steps and the length of each step taken. To comput the heading (the angle from the true north) we use direct cosin matrix to calculate the tilt angle. The computed angle used to orient each step to the corrected direction. To track and see the results the app used two methods, first the app store all the data in as CSV files the stored file have the x-y position for each step. Moreover, the app draw a real time graph to visualize the movement as a 2-D graph.

![pic2_2](https://user-images.githubusercontent.com/43111249/92306391-73caeb80-ef97-11ea-8b26-23fdb21e3294.png)

The app is divided to muliple packages:
1- Activity: this package contains three classes each represent a call to start a set of functions. The classes available are:
* UserListActivity: this is the initial class, you can added multiple users. By clicking on the username you will start next activity.
* UserActivity: to control each user that been created.
* GraphActivity: this is the main class which start positioning tracking and store all the data on the phone as CSV file.
2- Bias: it contains two class to calculate the bias from each sensor.

3- Dialog: this package contains an important dialogs for the app. The most important dialog is the calibration dialog which will going to give you an option to use the calibrated sensor (choosing auto) or the uncalibrated sensors (manually) to calibrate the sensor using our calibration algorithms.

4- Extra: this package contains an extra functions class which have some useful mathmatical functions. In this folder there are two Kalman filter class used in the previous version, but the one that have been used in the final version is in the OrientationFusionKalman package (more optimised).

5- Filewriting: in this package the classes used to write the CSV files in the device storage.

6- Filters: this package contains a three class for a common filters which are (low pass filter, mean filter and median filter).

7- Graph: this package contains the class for a real time graph to track and plot the user position.

8- Orientation: this package contains a main classes to compute the orientation from the gyroscope and magnetometer raw data.

9- OrientationFusedKalman: in this package there is a Kalman filter implementation and the complimantary filter as well. The filters worked based on fusion both of gyroscope orientation and magnetometer orientation.

10- Prefs: it contains the class responsable on choosing the prefered filter

11- StepCounting: this package contains the step counting algorithm as well as the stride length estimation.

12- FloorDetection: this package contains the floor detection algorithm.

### Results
The app settings used for optaining these results are Kalman filtered enable for fusion and for linear acceleration and low pass filter.

Figure 1: Linear acceleration before and after filtering
![FullTest_distance235 83](https://user-images.githubusercontent.com/43111249/92309766-02983200-efb1-11ea-9eda-88c17c1f2335.png)

Figure 2: The calculated tilt angle with and without filtering
![KalmanFilterHeading](https://user-images.githubusercontent.com/43111249/92309772-075ce600-efb1-11ea-9226-8eead077958f.png)

Figure 3: Walking for 235.8 meters and the drift between the starting and final point is less than 3 meters
![groundTruthAcc](https://user-images.githubusercontent.com/43111249/92309775-0cba3080-efb1-11ea-9646-d8b6ab3275b7.png)
