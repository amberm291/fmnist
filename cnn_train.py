import tensorflow as tf
import numpy as np
import glob
import re
import cv2
import cPickle as pickle
from sklearn import preprocessing
from data_set import dataSet

class trainNetwork:

    def __init__(self, train_dir, fname_img, fname_lbl):
        self.input_data = dataSet(train_dir)
        self.input_data.read_dataset(fname_img, fname_lbl)
        self.num_channels = self.input_data.CHANNELS

    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))


    def create_convolutional_layer(self, input_matrix,
                   num_input_channels,  conv_filter_size,        
                   num_filters, dropout_prob=None):  
        
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self.create_biases(num_filters)

        layer = tf.nn.conv2d(input=input_matrix,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

        layer += biases

        layer = tf.nn.max_pool(value=layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        layer = tf.nn.relu(layer)

        if dropout_prob:
            layer = tf.nn.dropout(layer, dropout_prob)
        return layer


    def create_flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])
        return layer


    def create_fc_layer(self, input_matrix,          
                 num_inputs, num_outputs,
                 use_relu=True, dropout_prob=None):
        
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)

        layer = tf.matmul(input_matrix, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        if dropout_prob:
            layer = tf.nn.dropout(layer, dropout_prob)

        return layer

    def build_network(self, x, filter_size_conv1, 
                filter_size_conv2, filter_size_conv3,
                num_filters, conv_dropout, fc_dropout,
                fc_layer_size):

        layer_conv1 = self.create_convolutional_layer(input_matrix=x,
                       num_input_channels=self.num_channels,
                       conv_filter_size=filter_size_conv1,
                       num_filters=num_filters)
                       #dropout_prob=conv_dropout)

        layer_conv2 = self.create_convolutional_layer(input_matrix=layer_conv1,
                       num_input_channels=num_filters,
                       conv_filter_size=filter_size_conv2,
                       num_filters=num_filters)
                       #dropout_prob=conv_dropout)

        layer_conv3 = self.create_convolutional_layer(input_matrix=layer_conv2,
                       num_input_channels=num_filters,
                       conv_filter_size=filter_size_conv3,
                       num_filters=2*num_filters)
                       #dropout_prob=conv_dropout)
                  
        layer_flat = self.create_flatten_layer(layer_conv3)

        layer_fc1 = self.create_fc_layer(input_matrix=layer_flat,
                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs=fc_layer_size,
                             use_relu=True,
                             dropout_prob=fc_dropout)

        layer_fc2 = self.create_fc_layer(input_matrix=layer_fc1,
                             num_inputs=fc_layer_size,
                             num_outputs=self.input_data.num_classes,
                             use_relu=False)

        return layer_fc2

    def train_network(self, filter_size_conv1, 
                filter_size_conv2, filter_size_conv3,
                num_filters, conv_dropout, fc_dropout,
                fc_layer_size, batch_size=50, epoch=50):

        x = tf.placeholder(tf.float32, shape=[None, self.input_data.HEIGHT*self.input_data.WIDTH*self.input_data.CHANNELS], name='x')
        x_image = tf.reshape(x, [-1, self.input_data.HEIGHT, self.input_data.WIDTH, self.input_data.CHANNELS])

        y_true = tf.placeholder(tf.float32, shape=[None, self.input_data.num_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)

        fc_layer = self.build_network(x_image, filter_size_conv1, 
                filter_size_conv2, filter_size_conv3,
                num_filters, conv_dropout, fc_dropout,
                fc_layer_size)


        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=fc_layer)

        cost = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        y_pred = tf.nn.softmax(fc_layer, name="y_pred")
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_iter = self.input_data.no_samples/batch_size
            for i in xrange(epoch):
                overall_pred = []
                overall_true = []
                for j in xrange(train_iter):
                    x_train, y_train = self.input_data.get_batch(batch_size)
                    feed_train = {x: x_train, y_true: y_train}
                    sess.run(optimizer, feed_dict=feed_train)
                    overall_pred.extend(list(y_pred_cls.eval({x: x_train, y_true: y_train})))
                    overall_true.extend(list(y_true_cls.eval({x: x_train, y_true: y_train})))
                print "epoch %d, training accuracy %f"%(i, self.calc_accuracy(overall_pred, overall_true))
                self.input_data.reset_index()
            saver.save(sess, "fmnist-diff-kernels")

    def calc_accuracy(self, overall_pred, overall_true):
        count = 0
        for i in xrange(len(overall_pred)):
            if overall_pred[i] == overall_true[i]: count += 1
        return float(count)/len(overall_pred)

if __name__=="__main__":
    training_folder = "../../data/fmnist/"
    fname_img = "train-images-idx3-ubyte"
    fname_lbl = "train-labels-idx1-ubyte"
    train_inst = trainNetwork(training_folder, fname_img, fname_lbl)
    filter_size = 3
    num_filters = 32
    conv_dropout = 0.7
    fc_dropout = 0.5
    fc_layer_size = 128
    train_inst.train_network(5, 4, 3, 
                        num_filters, conv_dropout, fc_dropout,
                        fc_layer_size)

