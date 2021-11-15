# Benchmarking ML and DL models on Mouse Movement Datasets for Continuous User Authentication

## Background and Motivation: ##

* Biometric systems use physical and
behavioral patterns such as face, fingerprint,
typing, swiping, and mouse movements for
user identification and authentication.

* Several studies have demonstrated that
mouse movements could be used for
continuous verification of users’ identity.

* The previous studies, however, are limited
in terms of the number of datasets, the
number of users, data collection scenarios,
and demonstrated high-error rates.

* Our research thus focuses on benchmarking
different ML and DL models on a variety of
datasets with the aim to provide a
comprehensive performance analysis of
mouse biometrics for continuous user
authentication.

## Continuous Authentication Framework: ##
Our training pipeline is similar to previous works, but modified to meet the requirements of each model. In general we serialized the data between each time intensive step, and all the data together in a single csv to reduce save and load times. 

### Machine Learning Models: ##
We extracted a total of 68 features for each user in each dataset. We used a feature selection algorithm SelectNonCollinear which took the best unique features for each user. Then, we performed hyperparameter optimization for HTER, a metric which combines the false accept and false reject rate,  on several ML Models, such as k-nearest neighbors, logistic regression, and support vector machine. We used the trained model with the optimal parameters given by hyperparameter optimization to predict whether given testing data was a genuine user or an imposter. 

### 1D-CNN: ###

Our 1D-CNN takes the mouse position as a 200x2 array. This is done by interpolating between each data point in the current window based on its time relative to the elapsed time of the window. This step is required to encode the time because the datasets were not collected at continuous time intervals. We cut any window that contained fewer than 20 data points to avoid training on windows where the mouse isn’t moving.

### 2D-CNN: ###

Our 2D-CNN requires 50x50 images. We use Bresinham's line algorithm to connect each data point in the time window to form a continuous path of mouse movement. An important note about this approach is that knowledge of the users’ monitor sizes are required. We were able to infer these values by looking at the max x-y position for each user and comparing them with common monitor sizes. The time relative to the start and end time of the time window is interpolated in the color channels. As with the 1D-CNN, any window with fewer than 10 data points was removed. 

## Pipeline: ###

## Models: ##
We chose our models in part based on [the 2020 paper]. Many of their models showed promising results for static authentication, so it was natural to adapt them for continuous authentication. It is important for our research to represent models from many ML paradigms. 

In training the models, the main hyperparameter that we consider is window size. The models are trained on 4 different window sizes-5, 10, 15, 20. 

### ML Models:
* KNN
* Gaussian Naive-Bayes
* Logistic regression
* Random forest
* SVM

 ### 1D-CNN: ###
Our 1D-CNN uses the almost same architecture as [the 2020 paper]. It is unique in that it has 2 convolutional input layers with kernel sizes of 10 and 15. The input layers feed into another convolutional layer with shared weights and a global max pooling layer. We add a fully connected layer with 100 neurons, and 'relu' activation function before the fully connected output layer.

### 2D-CNN: ### 
We use a similar architecture to the 2D-CNN in [the 2020 paper] with a few notable differences. First, we are using MobileNetV2 from tensorflow as the pretrained model instead of GoogLe Net. As with the 1D-CNN we add another fully connected layer before the output layer. 

### Training DL Models: ### 
Training both 1D and 2D-CNNs is somewhat involved because of the imbalanced classes. To combat this, we correct the initial bias  and class weights of the networks using the following equations: 

bias = log((size of class 1) / (size of class 0))

class 0 weight = 1 / (size of class 0) * (size of class 1 + size of class 0) / 2

class 1 weight = 1 / (size of class 1) * (size of class 1 + size of class 0) / 2

