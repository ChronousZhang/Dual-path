#coding:utf-8
#This function is used for split data to train/val/test
import numpy as  np 
import json 
import os
import shutil
from tqdm import tqdm 
import pandas as pd


def flickr30k():
    base_path = '/data/reid/flickr30k/'

    annotations = pd.read_table(os.path.join(base_path, 'results_20130124.token'), sep='\t', header=None, names=['image', 'caption'])

    #print(annotations)
    #print(annotations.keys())
    #print(annotations['image'])
    #print(annotations['caption'])
    #print(annotations.to_dict())
    data = dict()
    for image, text in zip(annotations['image'], annotations['caption']):
        image = image.split("#")[0]
        if image not in data:
            data[image] = [text]
        else:
            data[image].append(text)
    # print(data)
    #print(len(data))
    # dict is non-order
    name = list(data.keys())
    # split like Dual-Path
    train_name = name[:29783]
    val_name = name[29783:29783+1000]
    test_name = name[29783+1000:]
    assert len(train_name) == 29783
    assert len(val_name) == 1000
    assert len(test_name) == 1000
    train_data = dict()
    val_data = dict()
    test_data = dict()
    # each image is regard as one class
    # image2id = {_:i for i, _ in enumerate(name)}
    # print(image2id)
    for _ in name:
        image_path = os.path.join(base_path, 'flickr30k-images', _)
        # print(image_path)
        if _ in train_name:
            train_data[image_path] = data[_]
        elif _ in val_name:
            val_data[image_path] = data[_]
        elif _ in test_name:
            test_data[image_path] = data[_]
    #print(train_data)
    #print(val_data)
    #print(test_data)
    return train_data, val_data, test_data
        
# print(flickr30k)
