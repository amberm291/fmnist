import numpy as np
import glob
import re
import cv2
from sklearn import preprocessing
import gzip
import struct

class dataSet:
    def __init__(self, data_dir):
        self.current_index = 0
        self.no_samples = 0
        self.data_dir = data_dir

    def tokenize(self,filename):
        digits = re.compile(r'(\d+)')
        return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

    def read_dataset(self, fname_img, fname_lbl):
        self.num_classes = 10
        self.CHANNELS = 1
        with open(self.data_dir + fname_lbl) as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.labels = list(np.fromfile(flbl, dtype=np.int8))

        with open(self.data_dir + fname_img) as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.input_files = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols)

        for i in xrange(len(self.labels)):
            label_vector = [0]*10
            label_vector[self.labels[i]] = 1
            self.labels[i] = label_vector

        self.labels = np.array(self.labels)
        self.no_samples, self.HEIGHT, self.WIDTH = self.input_files.shape
        scaler = preprocessing.MinMaxScaler()
        self.vector_length = self.WIDTH*self.HEIGHT*self.CHANNELS
        self.input_files = self.input_files.reshape(self.no_samples, self.vector_length)
        scaler.fit(self.input_files)
        self.input_files = scaler.transform(self.input_files)
        perm = np.arange(len(self.labels))
        np.random.shuffle(perm)
        self.input_files = self.input_files[perm]
        self.labels = self.labels[perm]

    def create_input_matrix(self,input_folder):
        self.input_files = []
        self.labels = []
        self.num_classes = 3
        files = glob.glob(input_folder + "*")
        files.sort(key = self.tokenize)
        self.no_samples = len(files)
        print self.no_samples
        for i in xrange(self.no_samples):
            img = cv2.imread(files[i])
            img = cv2.resize(img,(128,128))
            self.input_files.append(img)
            if i/400 == 0:
                self.labels.append(np.array([1,0,0]))
            elif i/400 == 1:
                self.labels.append(np.array([0,1,0]))
            else:
                self.labels.append(np.array([0,0,1]))
        print img.shape
        self.HEIGHT, self.WIDTH, self.CHANNELS = img.shape
        self.input_files = np.array(self.input_files)
        scaler = preprocessing.MinMaxScaler()
        self.vector_length = self.WIDTH*self.HEIGHT*self.CHANNELS
        self.input_files = self.input_files.reshape(self.no_samples,self.vector_length)
        scaler.fit(self.input_files)
        self.input_files = scaler.transform(self.input_files)
        self.labels = np.array(self.labels)
        perm = np.arange(self.no_samples)
        np.random.shuffle(perm)
        self.input_files = self.input_files[perm]
        self.labels = self.labels[perm]

    def get_vector_length(self):
        return self.vector_length

    def get_batch(self, batch_size):
        if batch_size > self.no_samples: batch_size = self.no_samples
        start = self.current_index
        end = self.current_index + batch_size
        self.current_index += batch_size

        if self.current_index > self.no_samples:
            perm = np.arange(self.no_samples)
            np.random.shuffle(perm)
            self.input_files = self.input_files[perm]
            self.labels = self.labels[perm]
            start = 0
            end = batch_size
        
        return self.input_files[start:end], self.labels[start:end]

    def reset_index(self):
        self.current_index = 0
        perm = np.arange(self.no_samples)
        np.random.shuffle(perm)
        self.input_files = self.input_files[perm]
        self.labels = self.labels[perm]


if __name__=="__main__":
    training_folder = "../../data/fmnist/"
    fname_img = "train-images-idx3-ubyte"
    fname_lbl = "train-labels-idx1-ubyte"
    data_inst = dataSet(training_folder)
    data_inst.read_dataset(fname_img, fname_lbl)
