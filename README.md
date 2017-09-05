# Fashion MNIST Classifier

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is a data set developed by Zolando as a possible replacement for MNIST database. It contains 80k images of fashion apparels with 10 distinct classes. 

This repository contains the code to train a Convolutional Neural Network over the Fashion MNIST dataset in tensorflow. The overall architecture consists of 3 convolutional layers followed by one fully connected layer. Using successive convolution kernels of size 5, 4 and 3 repectively, test accuracy of 90.22% is achieved on this dataset.

The source contains only 3 files. `data_set.py` is a utility which reads the training and test samples. `cnn_train.py` is used to train the CNN on the training dataset and `test_network.py` is used to evaluate the trained network on the test datset.

The confusion matrix is as follows - 

| labels | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0      | 794   | 1     | 10    | 11    | 0     | 0     | 93    | 0     | 3     | 1     |
| 1      | 0     | 981   | 2     | 8     | 1     | 0     | 2     | 0     | 1     | 0     |
| 2      | 10    | 0     | 817   | 7     | 53    | 0     | 56    | 0     | 1     | 0     |
| 3      | 28    | 12    | 10    | 911   | 31    | 0     | 25    | 0     | 2     | 0     |
| 4      | 8     | 2     | 80    | 30    | 849   | 0     | 70    | 0     | 7     | 0     |
| 5      | 1     | 0     | 0     | 0     | 0     | 986   | 0     | 10    | 5     | 7     |
| 6      | 153   | 2     | 80    | 33    | 66    | 0     | 748   | 0     | 4     | 0     |
| 7      | 0     | 0     | 0     | 0     | 0     | 10    | 0     | 975   | 4     | 48    |
| 8      | 6     | 2     | 1     | 0     | 0     | 0     | 6     | 0     | 973   | 1     |
| 9      | 0     | 0     | 0     | 0     | 0     | 4     | 0     | 15    | 0     | 943   |

