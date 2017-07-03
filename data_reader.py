import numpy as np
import h5py
import os


class data_reader:
    def __init__(self):
        file_name = '/tempspace/hyuan/allen_data/training_new.h5'
        self.data = h5py.File(file_name, 'r')
        self.images = self.data['X']
        self.label = self.data['Y']
        self.train_range = 5000
        self.test_range = 1070
        self.train_idx = 0
        self.test_idx = 5000
        self.gen_index()
        print("DATA successfully loaded!!!!!!!!!========================")
    
    def gen_index(self):
        self.indexes = np.random.permutation(range(5000))
        self.train_idx = 0
    
    def next_batch(self, batch_size):
        next_index = self.train_idx + batch_size
        cur_indexes = list(self.indexes[self.train_idx:next_index])
        self.train_idx = next_index
        if len(cur_indexes) < batch_size:
            self.gen_index()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        return self.images[cur_indexes], self.label[cur_indexes]

    def next_test_batch(self, batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        return self.images[prev_idx:self.test_idx], self.label[prev_idx: self.test_idx]

    def reset(self):
        self.test_idx = 0
        
    def extract(self, imgs):
        return imgs[:,:,:,(0,2)]