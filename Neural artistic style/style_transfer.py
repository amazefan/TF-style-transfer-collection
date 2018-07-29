# -*- coding: utf-8 -*-
"""
Tensorflow Implentation of Style-transfer

@author: FAN
"""

import tensorflow as tf
import numpy as np
import os
import utils
from network import *

class Transfer:
    def __init__(self,content_img,style_img, loss_para = 0.001, pool_method = 'max',
                 content_layers = 'conv4_2', style_layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'],
                 epoches = 600,save_path = './dst'):
        self.C = content_img
        self.S = style_img
        self.loss_para = loss_para
        self.pool_method = pool_method
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.epoches = epoches
        self.save_path = save_path
        self._iter = 0
        self.timer = utils.Timer()
        
    def transfer(self):
        content = tf.placeholder(tf.float32)
        style = tf.placeholder(tf.float32)
        
        shape = self.C.shape
        with tf.variable_scope('train',reuse = tf.AUTO_REUSE):
            dst_img = tf.get_variable(name = 'dst_img',shape = shape, 
                                      initializer = tf.truncated_normal_initializer(stddev = 1.0))
        C = VGG19(self.C)
        C_net = C.forward_propogation(self.pool_method)
        S = VGG19(self.S)
        S_net = S.forward_propogation(self.pool_method)
        dst = VGG19(dst_img)
        dst_net = dst.forward_propogation(self.pool_method)
        
        global_step = tf.Variable(0,trainable=False)
        content_loss_in_dst = [dst_net[lay_name] for lay_name in self.content_layers]
        content_loss_in_C = [C_net[lay_name] for lay_name in self.content_layers]
        style_loss_in_dst = [dst_net[lay_name] for lay_name in self.style_layers]
        style_loss_in_S = [S_net[lay_name] for lay_name in self.style_layers]
        loss = self.content_loss(content_loss_in_dst,content_loss_in_C) + \
               tf.multiply(self.loss_para,self.style_loss(style_loss_in_dst,style_loss_in_S))
        optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': self.epoches})
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()
            print('Initialize finished.')
            optimizer1.minimize(sess,feed_dict={content:self.C, style:self.S},
                                fetches = [dst_img], loss_callback = self.call_back_func)
            '''
            for epoch in range(self.epoches):
                optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': self.epoches})
                if epoch%100 == 1:
                    img_data = dst_img.eval()
                    img_save = utils.reprocess_img(img_data)
                    utils.save_img(img_save,save_dir= self.save_path,epoch = epoch)
            '''
            
            #utils.save_img(utils.reprocess_img(dst_img.eval()),save_dir= self.save_path,epoch = 'full')
            utils.save_img_PIL(dst_img.eval(),save_dir= self.save_path,epoch = 'full')
            #return utils.reprocess_img(dst_img.eval())
 
    def call_back_func(self,dst_img):
        self.timer.time_use()
        if self._iter%200 ==1:
            #img_save = utils.reprocess_img(dst_img)
            #utils.save_img(img_save,save_dir= self.save_path,epoch = self._iter)
            utils.save_img_PIL(dst_img,save_dir= self.save_path,epoch = self._iter)
        elif self._iter%5 ==0:   
            print('----------------------------')
            print('Now is iter {}.'.format(self._iter))
            print('Use time {}.'.format(self.timer.time_formated))
            print('This five iteration use time {:.1f}sec.'.format(self.timer.time_used*5))
            print('Expected to finished training in {}.'.format(self.timer.expect_time(self._iter,self.epoches)))
            print('----------------------------')
        self._iter += 1
           
    @staticmethod
    def content_loss(dst_layer,content_layer):
        if isinstance(dst_layer,list):
            for dst_value,content_value in zip(dst_layer,content_layer):
                content_loss_in_layer = tf.reduce_mean(tf.square(dst_value-content_value))
                tf.add_to_collection('content_loss',content_loss_in_layer)
            content_loss_all = tf.add_n(tf.get_collection('content_loss'))  
        else:
            content_loss_all = tf.reduce_mean(tf.square(dst_layer-content_layer))           
        return content_loss_all
            
    def style_loss(self,dst_layer,style_layer):
        for dst_value,style_value in zip(dst_layer,style_layer):
            shape = dst_value.get_shape()
            Nl = int(shape[3])
            Ml = int(shape[1]*shape[2])
            para = 1/(4*(Nl**2)*(Ml**2))*(Nl**2)
            Gram_dst = self.Gram(dst_value)
            Gram_style = self.Gram(style_value)
            Gram_loss_all = tf.reduce_mean(tf.square(Gram_dst-Gram_style))  
            style_loss_in_layer = tf.multiply(para,Gram_loss_all)
            tf.add_to_collection('style_loss',style_loss_in_layer)
        style_loss = tf.add_n(tf.get_collection('style_loss'))
        return style_loss
            
    @staticmethod
    def Gram(input_feat):
        shape = input_feat.get_shape()
        num_channels = int(shape[3])
        
        feat_matrix = tf.reshape(input_feat,shape = [-1,num_channels])
        
        Gram_matrix = tf.matmul(tf.transpose(feat_matrix),feat_matrix)
        
        return Gram_matrix
        
    
    
    