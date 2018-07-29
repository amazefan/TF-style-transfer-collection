# -*- coding: utf-8 -*-
"""
Tensorflow Implentation of Style-transfer

@author: FAN
"""
import tensorflow as tf
import numpy as np
import os 
os.chdir(r'./model/')

vgg_19_without_fc = np.load('vgg19-notop.npy').tolist()

#数据结构类型：字典，key值为conv层，一个conv层两个index，分别为weights和biases
conv_layer_name = [layer for layer in vgg_19_without_fc]
conv_layer_weights = [vgg_19_without_fc[layer][0] for layer in vgg_19_without_fc]
conv_layer_biases = [vgg_19_without_fc[layer][1] for layer in vgg_19_without_fc]

class VGG19:
    struct = ['conv1_1','relu1_1','conv1_2','relu1_2','pool1',
              'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
              'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
              'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
              'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5']
    
    vgg_19_without_fc = np.load('vgg19-notop.npy').tolist()
    conv_layer_name = [layer for layer in vgg_19_without_fc]
    conv_layer_name.sort()
    conv_layer_weights = [vgg_19_without_fc[layer][0] for layer in conv_layer_name]
    conv_layer_biases = [vgg_19_without_fc[layer][1] for layer in conv_layer_name]
    #mean_ = (103.939, 116.779, 123.68)
    
    def __init__(self,input_):
        shape_rank = len(input_.shape)
        if shape_rank == 3:
            #input_-=self.mean_
            input_ = input_[np.newaxis,:]
        #CV2读入数据格式是BGR，模型默认训练顺序是RGB
        #input_ = input_[:,:,:,::-1]
        self.input = input_
        
    def forward_propogation(self, pool_method = 'max'):
        #起始模块

        net = {}
        kernel1_1,bias1_1 = tf.constant(self.conv_layer_weights[0]),tf.constant(self.conv_layer_biases[0])
        conv1_1 = self.conv_layer(self.input,kernel1_1)
        net.setdefault('conv1_1',conv1_1)
        relu1_1 = self.relu_layer(conv1_1,bias1_1)
        net.setdefault('relu1_1',relu1_1)
        kernel1_2,bias1_2 = tf.constant(self.conv_layer_weights[1]),tf.constant(self.conv_layer_biases[1])
        conv1_2 = self.conv_layer(relu1_1,kernel1_2)
        net.setdefault('conv1_2',conv1_2)
        relu1_2 = self.relu_layer(conv1_2,bias1_2)
        net.setdefault('relu1_2',relu1_2)
        pool1 = self.pool_layer(relu1_2,pool_method)
        net.setdefault('pool1',pool1)
        
        #第二个模块
        kernel2_1,bias2_1 = tf.constant(self.conv_layer_weights[2]),tf.constant(self.conv_layer_biases[2])
        conv2_1 = self.conv_layer(pool1,kernel2_1)
        net.setdefault('conv2_1',conv2_1)
        relu2_1 = self.relu_layer(conv2_1,bias2_1)
        net.setdefault('relu2_1',relu2_1)
        kernel2_2,bias2_2 = tf.constant(self.conv_layer_weights[3]),tf.constant(self.conv_layer_biases[3])
        conv2_2 = self.conv_layer(relu2_1,kernel2_2)
        net.setdefault('conv2_2',conv2_2)
        relu2_2 = self.relu_layer(conv2_2,bias2_2)
        net.setdefault('relu2_2',relu2_2)
        pool2 = self.pool_layer(relu2_2,pool_method)
        net.setdefault('pool2',pool2)
        
        #第3个模块     
        kernel3_1,bias3_1 = tf.constant(self.conv_layer_weights[4]),tf.constant(self.conv_layer_biases[4])
        conv3_1 = self.conv_layer(pool2,kernel3_1)
        net.setdefault('conv3_1',conv3_1)
        relu3_1 = self.relu_layer(conv3_1,bias3_1)
        net.setdefault('relu3_1',relu3_1)
        kernel3_2,bias3_2 = tf.constant(self.conv_layer_weights[5]),tf.constant(self.conv_layer_biases[5])
        conv3_2 = self.conv_layer(relu3_1,kernel3_2)
        net.setdefault('conv3_2',conv3_2)
        relu3_2 = self.relu_layer(conv3_2,bias3_2)
        net.setdefault('relu3_2',relu3_2)
        kernel3_3,bias3_3 = tf.constant(self.conv_layer_weights[6]),tf.constant(self.conv_layer_biases[6])
        conv3_3 = self.conv_layer(relu3_2,kernel3_3)
        net.setdefault('conv3_3',conv3_3)
        relu3_3 = self.relu_layer(conv3_3,bias3_3)
        net.setdefault('relu3_3',relu3_3)
        kernel3_4,bias3_4 = tf.constant(self.conv_layer_weights[7]),tf.constant(self.conv_layer_biases[7])
        conv3_4 = self.conv_layer(relu3_3,kernel3_4)
        net.setdefault('conv3_4',conv3_4)
        relu3_4 = self.relu_layer(conv3_4,bias3_4)
        net.setdefault('relu3_4',relu3_4)
        pool3 = self.pool_layer(relu3_4,pool_method)
        net.setdefault('pool3',pool3)
        
        #第4个模块
        kernel4_1,bias4_1 = tf.constant(self.conv_layer_weights[8]),tf.constant(self.conv_layer_biases[8])
        conv4_1 = self.conv_layer(pool3,kernel4_1)
        net.setdefault('conv4_1',conv4_1)
        relu4_1 = self.relu_layer(conv4_1,bias4_1)
        net.setdefault('relu4_1',relu4_1)
        kernel4_2,bias4_2 = tf.constant(self.conv_layer_weights[9]),tf.constant(self.conv_layer_biases[9])
        conv4_2 = self.conv_layer(relu4_1,kernel4_2)
        net.setdefault('conv4_2',conv4_2)
        relu4_2 = self.relu_layer(conv4_2,bias4_2)
        net.setdefault('relu4_2',relu4_2)
        kernel4_3,bias4_3 = tf.constant(self.conv_layer_weights[10]),tf.constant(self.conv_layer_biases[10])
        conv4_3 = self.conv_layer(relu4_2,kernel4_3)
        net.setdefault('conv4_3',conv4_3)
        relu4_3 = self.relu_layer(conv4_3,bias4_3)
        net.setdefault('relu4_3',relu4_3)
        kernel4_4,bias4_4 = tf.constant(self.conv_layer_weights[11]),tf.constant(self.conv_layer_biases[11])
        conv4_4 = self.conv_layer(relu4_3,kernel4_4)
        net.setdefault('conv4_4',conv4_4)
        relu4_4 = self.relu_layer(conv4_4,bias4_4)
        net.setdefault('relu4_4',relu4_4)
        pool4 = self.pool_layer(relu4_4,pool_method)
        net.setdefault('pool4',pool4)
        
        #第5个模块 
        kernel5_1,bias5_1 = tf.constant(self.conv_layer_weights[12]),tf.constant(self.conv_layer_biases[12])
        conv5_1 = self.conv_layer(pool4,kernel5_1)
        net.setdefault('conv5_1',conv5_1)
        relu5_1 = self.relu_layer(conv5_1,bias5_1)
        net.setdefault('relu5_1',relu5_1)
        kernel5_2,bias5_2 = tf.constant(self.conv_layer_weights[13]),tf.constant(self.conv_layer_biases[13])
        conv5_2 = self.conv_layer(relu5_1,kernel5_2)
        net.setdefault('conv5_2',conv5_2)
        relu5_2 = self.relu_layer(conv5_2,bias5_2)
        net.setdefault('relu5_2',relu5_2)
        kernel5_3,bias5_3 = tf.constant(self.conv_layer_weights[14]),tf.constant(self.conv_layer_biases[14])
        conv5_3 = self.conv_layer(relu5_2,kernel5_3)
        net.setdefault('conv5_3',conv5_3)
        relu5_3 = self.relu_layer(conv5_3,bias5_3)
        net.setdefault('relu5_3',relu5_3)
        kernel5_4,bias5_4 = tf.constant(self.conv_layer_weights[15]),tf.constant(self.conv_layer_biases[15])
        conv5_4 = self.conv_layer(relu5_3,kernel5_4)
        net.setdefault('conv5_4',conv5_4)
        relu5_4 = self.relu_layer(conv5_4,bias5_4)
        net.setdefault('relu5_4',relu5_4)
        pool5 = self.pool_layer(relu5_4,pool_method)
        net.setdefault('pool5',pool5)
        
        return net
    
    def submodel(self,sub_id):
        #sub_model选项，子模型：first,second,third,fourth,fifth代表五个模块
        if sub_id == 'first':
            kernel1_1,bias1_1 = tf.constant(self.conv_layer_weights[0]),tf.constant(self.conv_layer_biases[0])
            conv1_1 = self.conv_layer(self.input,kernel1_1)
            relu1_1 = self.relu_layer(conv1_1,bias1_1)
            kernel1_2,bias1_2 = tf.constant(self.conv_layer_weights[1]),tf.constant(self.conv_layer_biases[1])
            conv1_2 = self.conv_layer(relu1_1,kernel1_2)
            relu1_2 = self.relu_layer(conv1_2,bias1_2)
            pool1 = self.pool_layer(relu1_2)
            kernel2_1,bias2_1 = tf.constant(self.conv_layer_weights[2]),tf.constant(self.conv_layer_biases[2])
            conv2_1 = self.conv_layer(pool1,kernel2_1)
            relu2_1 = self.relu_layer(conv2_1,bias2_1)
            
            return relu2_1
        
        if sub_id == 'second':
            kernel2_1,bias2_1 = tf.constant(self.conv_layer_weights[2]),tf.constant(self.conv_layer_biases[2])
            conv2_1 = self.conv_layer(self.input,kernel2_1)
            relu2_1 = self.relu_layer(conv2_1,bias2_1)
            kernel2_2,bias2_2 = tf.constant(self.conv_layer_weights[3]),tf.constant(self.conv_layer_biases[3])
            conv2_2 = self.conv_layer(relu2_1,kernel2_2)
            relu2_2 = self.relu_layer(conv2_2,bias2_2)
            pool2 = self.pool_layer(relu2_2)
            kernel3_1,bias3_1 = tf.constant(self.conv_layer_weights[4]),tf.constant(self.conv_layer_biases[4])
            conv3_1 = self.conv_layer(pool2,kernel3_1)
            relu3_1 = self.relu_layer(conv3_1,bias3_1)
            
            return relu3_1
        
        if sub_id == 'third':
            kernel3_1,bias3_1 = tf.constant(self.conv_layer_weights[4]),tf.constant(self.conv_layer_biases[4])
            conv3_1 = self.conv_layer(self.input,kernel3_1)
            relu3_1 = self.relu_layer(conv3_1,bias3_1)
            kernel3_2,bias3_2 = tf.constant(self.conv_layer_weights[5]),tf.constant(self.conv_layer_biases[5])
            conv3_2 = self.conv_layer(relu3_1,kernel3_2)
            relu3_2 = self.relu_layer(conv3_2,bias3_2)
            kernel3_3,bias3_3 = tf.constant(self.conv_layer_weights[6]),tf.constant(self.conv_layer_biases[6])
            conv3_3 = self.conv_layer(relu3_2,kernel3_3)
            relu3_3 = self.relu_layer(conv3_3,bias3_3)
            kernel3_4,bias3_4 = tf.constant(self.conv_layer_weights[7]),tf.constant(self.conv_layer_biases[7])
            conv3_4 = self.conv_layer(relu3_3,kernel3_4)
            relu3_4 = self.relu_layer(conv3_4,bias3_4)
            pool3 = self.pool_layer(relu3_4)
            kernel4_1,bias4_1 = tf.constant(self.conv_layer_weights[8]),tf.constant(self.conv_layer_biases[8])
            conv4_1 = self.conv_layer(pool3,kernel4_1)
            relu4_1 = self.relu_layer(conv4_1,bias4_1)
            
            return relu4_1
        
        if sub_id == 'fourth':
            kernel4_1,bias4_1 = tf.constant(self.conv_layer_weights[8]),tf.constant(self.conv_layer_biases[8])
            conv4_1 = self.conv_layer(self.input,kernel4_1)
            relu4_1 = self.relu_layer(conv4_1,bias4_1)
            kernel4_2,bias4_2 = tf.constant(self.conv_layer_weights[9]),tf.constant(self.conv_layer_biases[9])
            conv4_2 = self.conv_layer(relu4_1,kernel4_2)
            relu4_2 = self.relu_layer(conv4_2,bias4_2)
            kernel4_3,bias4_3 = tf.constant(self.conv_layer_weights[10]),tf.constant(self.conv_layer_biases[10])
            conv4_3 = self.conv_layer(relu4_2,kernel4_3)
            relu4_3 = self.relu_layer(conv4_3,bias4_3)
            kernel4_4,bias4_4 = tf.constant(self.conv_layer_weights[11]),tf.constant(self.conv_layer_biases[11])
            conv4_4 = self.conv_layer(relu4_3,kernel4_4)
            relu4_4 = self.relu_layer(conv4_4,bias4_4)
            pool4 = self.pool_layer(relu4_4)
            kernel5_1,bias5_1 = tf.constant(self.conv_layer_weights[12]),tf.constant(self.conv_layer_biases[12])
            conv5_1 = self.conv_layer(pool4,kernel5_1)
            relu5_1 = self.relu_layer(conv5_1,bias5_1)
            
            return relu5_1
     
    def trueresult(self,sub_id):
        if sub_id == 'first':
            kernel1_1,bias1_1 = tf.constant(self.conv_layer_weights[0]),tf.constant(self.conv_layer_biases[0])
            conv1_1 = self.conv_layer(self.input,kernel1_1)
            relu1_1 = self.relu_layer(conv1_1,bias1_1)
            
            return relu1_1
     
        if sub_id == 'second':
            kernel2_1,bias2_1 = tf.constant(self.conv_layer_weights[2]),tf.constant(self.conv_layer_biases[2])
            conv2_1 = self.conv_layer(self.input,kernel2_1)
            relu2_1 = self.relu_layer(conv2_1,bias2_1)
            
            return relu2_1
        
        if sub_id == 'third':
            kernel3_1,bias3_1 = tf.constant(self.conv_layer_weights[4]),tf.constant(self.conv_layer_biases[4])
            conv3_1 = self.conv_layer(self.input,kernel3_1)
            relu3_1 = self.relu_layer(conv3_1,bias3_1)

            return relu3_1

        if sub_id == 'fourth':
            kernel4_1,bias4_1 = tf.constant(self.conv_layer_weights[8]),tf.constant(self.conv_layer_biases[8])
            conv4_1 = self.conv_layer(self.input,kernel4_1)
            relu4_1 = self.relu_layer(conv4_1,bias4_1)            
        
            return relu4_1
            
    @staticmethod    
    def conv_layer(in_tensor,kernel):
        return tf.nn.conv2d(in_tensor,kernel,strides = [1,1,1,1],padding = 'SAME')
    
    @staticmethod
    def relu_layer(in_tensor,bias):
        return tf.nn.relu(tf.nn.bias_add(in_tensor,bias))
    
    @staticmethod
    def pool_layer(in_tensor,pool_method = 'max'):
        if pool_method == 'max':
            return tf.nn.max_pool(in_tensor,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
        elif pool_method == 'avg':
            return tf.nn.avg_pool(in_tensor,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
        
        