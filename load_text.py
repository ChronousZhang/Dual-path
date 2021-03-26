#coding:utf-8
import numpy as np 
import os 
import torch
import torch.nn as nn 
import random 
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

def load_data(dataset_path,part,batch_size, image_dataset):
    all_path = []
    data_path = os.path.join(dataset_path,part)
    singal = os.listdir(data_path)
    # print(singal)
    for fsingal in singal:
        filepath = os.path.join(data_path,fsingal)
        filename = os.listdir(filepath)
        # print(fsingal, filename)
        # print("========================")
        for fname in filename:
            ffpath = os.path.join(filepath, fname)
            # associate image label and text label
            # print(fsingal, image_dataset.class_to_idx, image_dataset.class_to_idx[str(fsingal)])
            path = [image_dataset.class_to_idx[str(fsingal)], ffpath]
            # only get text file but not image 
            if '.npy' in path[1]:
                all_path.append(path)
        
    count = len(all_path)
    data_x = np.empty((count,300,1,56),dtype='float32') # 300 is each word's word2vector length, 56 is length of sequence
    data_y = []

    random.shuffle(all_path)
    #print(all_path)
    for i, item in enumerate(all_path):
        text = np.load(item[1])
        # print(text.shape)  # shape: 56*300
        text_a = text[np.newaxis,:]
        text_b = text_a.transpose((2,0,1))

        data_x[i,:,:,:] = text_b
        data_y.append(int(item[0]))
        #print(item)

    data_y = np.asarray(data_y)

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    #print(data_y)
    dataset = dataf.TensorDataset(data_x,data_y)
    loader = dataf.DataLoader(dataset,batch_size,shuffle=True,num_workers=16,drop_last=False)
    print("Load Text Data Done##")
    return loader

def load_dataset(dataset_path,part,batch_size, image_dataset):
    all_path = []
    data_path = os.path.join(dataset_path,part)
    singal = os.listdir(data_path)
    # print(singal)
    for fsingal in singal:
        filepath = os.path.join(data_path,fsingal)
        filename = os.listdir(filepath)
        # print(fsingal, filename)
        # print("========================")
        for fname in filename:
            ffpath = os.path.join(filepath, fname)
            # associate image label and text label
            # print(fsingal, image_dataset.class_to_idx, image_dataset.class_to_idx[str(fsingal)])
            path = [image_dataset.class_to_idx[str(fsingal)], ffpath]
            # only get text file but not image 
            if '.npy' in path[1]:
                all_path.append(path)
        
    count = len(all_path)
    data_x = np.empty((count,300,1,56),dtype='float32') # 300 is each word's word2vector length, 56 is length of sequence
    data_y = []

    random.shuffle(all_path)
    #print(all_path)
    for i, item in enumerate(all_path):
        text = np.load(item[1])
        # print(text.shape)  # shape: 56*300
        text_a = text[np.newaxis,:]
        text_b = text_a.transpose((2,0,1))

        data_x[i,:,:,:] = text_b
        data_y.append(int(item[0]))

    data_y = np.asarray(data_y)

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    dataset = dataf.TensorDataset(data_x,data_y)
    del data_x
    del data_y
    del all_path
    del singal
    print("Load Text Data Done##")
    return dataset

if __name__ == "__main__":
    get = load_data('./data_test','val',8)
    print('get_length:: ',len(get))
    print(len(get.dataset))
    print('get_type：',type(get))
    print(len(get.dataset))
    for data,label in get:
        print("data:  ",data.shape)
        print("label:  ",label.data)
