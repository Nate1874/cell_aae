import numpy as np
import h5py
import os


class data_reader:
    def __init__(self):
        file_name = '/tempspace/hyuan/allen_data/training.h5'
        self.data = h5py.File(file_name, 'r')
        self.images = self.data['X'][0:5000, :, :, :]
        self.label = self.data['Y'][0:5000, :]
        self.image_test = self.data['X'][5000:, :, :, :]
        self.label_test = self.data['Y'][5000:, :]
        self.train_idx = 0
        self.test_idx = 0
        print("DATA successfully loaded!!!!!!!!!========================")

    def next_batch(self, batch_size):
        prev_idx = self.train_idx
        self.train_idx += batch_size
        if self.train_idx > self.images.shape[0]:
            self.train_idx = batch_size
            prev_idx = 0
        return self.images[prev_idx:self.train_idx, :, :, :], self.label[
            prev_idx:self.train_idx, :]

    def next_test_batch(self, batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        return self.image_test[
            prev_idx:self.test_idx, :, :, :], self.label_test[prev_idx:
                                                              self.test_idx, :]

    def reset(self):
        self.test_idx = 0
        
    def extract(self, imgs):
        return imgs[:,:,:,(0,2)]