# -*- coding: utf-8 -*-
"""
Tensorflow Implentation of Style-transfer

@author: FAN
"""

import numpy as np
import cv2
import os
from time import time
import PIL.Image

def get_img(path, max_size = None):
    img = cv2.imread(path)
    if max_size is not None:
        factor = max_size/np.max(img.shape[0:-1])
        shape = np.array(img.shape[0:-1])*factor
        size = shape.astype(int)
        img = cv2.resize(img,tuple(size[::-1]))
    return np.float32(img)

def get_img_PIL(path, max_size = None):
    img = PIL.Image.open(path)
    if max_size is not None:
        factor = float(max_size)/np.max(img.size)
        shape = np.array(img.size)*factor
        size = shape.astype(int)
        img = img.resize(size,PIL.Image.LANCZOS)
    return np.float32(img)

def preprocess_img(img_data):
    shape_rank = len(img_data.shape)
    if shape_rank !=4:
        img_data_new = img_data[np.newaxis,:]
    else:img_data_new = img_data       
    return img_data_new

def same_shape(img_A,img_B):
    h_min = np.min((img_A.shape[0],img_B.shape[0]))
    w_min = np.min((img_A.shape[1],img_B.shape[1]))
    same_A = cv2.resize(img_A,(w_min,h_min))
    same_B = cv2.resize(img_B,(w_min,h_min))
    return same_A,same_B

def transition_img(tensor):
    t2p = tensor.eval()
    t2p_1 = t2p[:,:,:,0]
    save = np.clip(t2p_1 + 127 , 0,255)
    return save

   
def reprocess_img(img_data):
    img_data_processed = np.clip(img_data,0,255)
    img_data_processed = np.uint8(img_data_processed)
    img_data_processed = img_data_processed[:,:,::-1]
    return img_data_processed

def save_img(img_data,save_dir,epoch):
    full_path = os.path.join(save_dir ,'style-transfer_{}.jpg'.format(epoch))
    cv2.imwrite(full_path,img_data)
    print('Sucessfully saved and named {} with epoch {}.'.format('style-transfer',epoch))

def save_img_PIL(img_data,save_dir,epoch):
    full_path = os.path.join(save_dir ,'style-transfer_{}.jpg'.format(epoch))
    image = np.clip(img_data, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(full_path, 'wb') as file:
        PIL.Image.fromarray(image).save(file)
    print('Sucessfully saved and named {} with epoch {}.'.format('style-transfer',epoch))


#save_dir = r'E:\python.file\git-Repository\TF--style-transfer-collection\dst'
#save_img(img,save_dir)

class Timer:
    def __init__(self):
        self.begin = time()
        self.last_iter = time()
        self.time_all = 0
        
    def time_use(self):
        self.now = time()
        self.time_used = self.now - self.last_iter
        self.last_iter = self.now
        self.time_all += self.time_used
        self.time_formated = '{}h {}min {:.1f}sec'.format(self.time_all//3600,(self.time_all%3600)//60,(self.time_all%3600)%60)
    
    def expect_time(self,_iter,epoches):
        expected = self.time_all/(_iter+1)*(epoches - _iter)
        finish_time = '{:.1f}h {:.1f}min {:.1f}sec'.format(expected//3600,(expected%3600)//60,(expected%3600)%60)
        return finish_time
        

