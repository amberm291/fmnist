import tensorflow as tf
import numpy as np
import glob
import re
import cv2
import cPickle as pickle
from sklearn import preprocessing
from data_set import dataSet

def calc_accuracy(overall_pred, overall_true):
    count = 0
    for i in xrange(len(overall_pred)):
        if overall_pred[i] == overall_true[i]: count += 1
    return float(count)/len(overall_pred)

def confusion_matrix(overall_pred, overall_true):
    count_dict = {}
    for i in xrange(len(overall_true)):
        true_label = overall_true[i]
        pred_label = overall_pred[i]
        if true_label not in count_dict: count_dict[true_label] = {}
        if pred_label not in count_dict[true_label]: count_dict[true_label][pred_label] = 0
        count_dict[true_label][pred_label] += 1

    for true_label in count_dict:
        print true_label, count_dict[true_label]

if __name__=="__main__":
    train_dir = "../../data/fmnist/"
    fname_img = "t10k-images-idx3-ubyte"
    fname_lbl = "t10k-labels-idx1-ubyte"
    input_data = dataSet(train_dir)
    input_data.read_dataset(fname_img, fname_lbl)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('fmnist-same-kernels.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_true_cls = tf.argmax(y_true, dimension=1)
        y_pred = graph.get_tensor_by_name("y_pred:0")
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        test_samples, test_labels = input_data.get_batch(input_data.no_samples)
        print test_samples.shape
        feed_test = {x: test_samples, y_true: test_labels}
        overall_pred = list(y_pred_cls.eval(feed_test))
        overall_true = list(y_true_cls.eval(feed_test))
        print confusion_matrix(overall_pred, overall_true)