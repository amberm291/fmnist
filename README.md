# Fashion MNIST Classifier

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is a data set developed by Zolando as a possible replacement for MNIST database. It contains 80k images of fashion apparels with 10 distinct classes. 

This repository contains the code to train a Convolutional Neural Network over the Fashion MNIST dataset in tensorflow. The overall architecture consists of 3 convolutional layers followed by one fully cionnected layer. Using successive convolution kernel size of 5, 4 and 3 test acciracy of 90.22% is achieved on this dataset.

The source contains only 3 files. `data_set.py` is a utility which reads the training and test samples. `cnn_train.py` is used to train the CNN on the training dataset and `test_network.py` is used to evaluate the trained network on the test datset.