# -*- coding: utf-8 -*-
"""
Tensorflow style-transfer collection

@author: Fan
"""

import tensorflow as tf
import numpy as np
import utils
from NNF_func import *
from network import *
import cv2

class VisualMapping:
    def __init__(self,map_img, mapped_img, epoches = 600, alpha = [0.8,0.7,0.6,0.1]):
        self.A = mapped_img
        self.B = map_img
        self.epoches = epoches  
        self.alpha = alpha
        self.timer = utils.Timer()
        
    def get_feat(self):
        MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((1,1,3))
        
        img_A = (self.A-MEAN_VALUES).astype('float32')
        A_model = VGG19(img_A)
        A_layers = A_model.forward_propogation()
        A_feat = [A_layers['relu5_1'],A_layers['relu4_1'],A_layers['relu3_1'],A_layers['relu2_1'],A_layers['relu1_1']]
        
        img_B = (self.B-MEAN_VALUES).astype('float32')
        B_model = VGG19(img_B)
        B_layers = B_model.forward_propogation()
        B_feat = [B_layers['relu5_1'],B_layers['relu4_1'],B_layers['relu3_1'],B_layers['relu2_1'],B_layers['relu1_1']]
        
        self.A_feat = A_feat
        self.B_feat = B_feat
    
    def NNF_computing(self,sess):
        with sess.as_default():
            order = ['fourth','third','second','first']
            layer = [256,128,64,3]
            for i in range(len(self.B_feat)):
                print('Processing Layer {}'.format(5-i))
                if i<=2:
                    win_rad = 1
                else:win_rad = 2
                
                Al  = self.A_feat[i].eval()
                Al  = np.squeeze(Al)
                Al_norm  = Al/np.sqrt(np.sum(Al**2,axis = (0,1),keepdims = True))
                if i==0:
                    Al_m_norm = Al_norm.copy()
                    Al_m = Al.copy()
                
                Bl_m = self.B_feat[i].eval()
                Bl_m = np.squeeze(Bl_m)
                Bl_m_norm = Bl_m/np.sqrt(np.sum(Bl_m**2,axis = (0,1),keepdims = True))
                if i==0:
                    Bl_norm = Bl_m_norm.copy()
                    Bl = Bl_m.copy()
                
                A_shape = Al.shape[0:-1]
                print(A_shape)
                B_shape = Bl_m.shape[0:-1]
                print(B_shape)
                
                if i==0 :
                    iter_time = 4
                    nnf_init_A = initialize_nnf(A_shape,B_shape)
                    nnf_value_init_A = initialize_nnf_value(nnf_init_A,Al_norm,Al_m_norm,Bl_norm,Bl_m_norm,win_rad)
                    
                    nnf_init_Bm = initialize_nnf(B_shape,A_shape)
                    nnf_value_init_Bm = initialize_nnf_value(nnf_init_Bm,Bl_m_norm,Bl_norm,Al_m_norm,Al_norm,win_rad)
                
                else:
                    iter_time = 3
                    nnf_init_A = cv2.resize(nnf_A*2,A_shape[::-1],interpolation = cv2.INTER_NEAREST)
                    nnf_value_init_A = initialize_nnf_value(nnf_init_A,Al_norm,Al_m_norm,Bl_norm,Bl_m_norm,win_rad)
                    nnf_init_Bm = cv2.resize(nnf_B*2,B_shape[::-1],interpolation = cv2.INTER_NEAREST)
                    nnf_value_init_Bm = initialize_nnf_value(nnf_init_Bm,Bl_m_norm,Bl_norm,Al_m_norm,Al_norm,win_rad)
                    
                nnf_A,nnf_value_A = propogation(nnf_init_A,nnf_value_init_A,Al_norm,Al_m_norm,Bl_norm,Bl_m_norm,win_rad,iter_time)            
                warped_A = reconstruct(Al,Bl_m,nnf_A)
                print(warped_A.shape)

                nnf_B,nnf_value_Bm = propogation(nnf_init_Bm,nnf_value_init_Bm,Bl_m_norm,Bl_norm,Al_m_norm,Al_norm,win_rad,iter_time)            
                warped_Bm = reconstruct(Bl_m,Al,nnf_B)
                print(warped_Bm.shape)
                
                if i<4:
                    with tf.variable_scope('L{}'.format(5-i),reuse = tf.AUTO_REUSE):
                        Al_relu41  = tf.get_variable('Ra',shape = self.A_feat[i+1].get_shape().as_list()[0:-1] + [layer[i]],dtype = tf.float32,
                                                initializer = tf.truncated_normal_initializer(stddev=1.0))
                        Blm_relu41 = tf.get_variable('Rbm',shape = self.B_feat[i+1].get_shape().as_list()[0:-1] + [layer[i]],dtype = tf.float32,
                                                initializer = tf.truncated_normal_initializer(stddev=1.0))
                    
                    R_Al  = VGG19(Al_relu41).submodel(sub_id = order[i])
                    R_Blm = VGG19(Blm_relu41).submodel(sub_id = order[i])
                    print(R_Al.get_shape())
                    print(R_Blm.get_shape())                    
                    
                    loss_A  = self.sqrloss(warped_A, R_Al)
                    loss_Bm = self.sqrloss(warped_Bm, R_Blm)
                    optimizerA = tf.contrib.opt.ScipyOptimizerInterface(loss_A, method='L-BFGS-B', options={'maxiter': self.epoches})
                    optimizerB = tf.contrib.opt.ScipyOptimizerInterface(loss_Bm, method='L-BFGS-B', options={'maxiter': self.epoches})
                    
                    #train_opA = optimizerA.minimize(loss_A)
                    #train_opBm = optimizerB.minimize(loss_Bm)
                    
                    holder1= tf.placeholder(tf.float32)
                    holder2= tf.placeholder(tf.float32)
                
                    tf.global_variables_initializer().run()
                    '''
                    for j in range(self.epoches):
                        sess.run(train_opA)
                        sess.run(train_opBm)
                        if j % 200==0:
                          curr_lossA = loss_A.eval()
                          curr_lossBm = loss_Bm.eval()
                          print("At iterate {}\tf=  {}".format(j, curr_lossA))
                          print("At iterate {}\tf=  {}".format(j, curr_lossBm))
                    '''
                    self._iter = 0
                    optimizerA.minimize(sess,feed_dict={holder1:warped_A},
                                fetches = [loss_A], loss_callback = self.callback)
                    self._iter = 0
                    optimizerB.minimize(sess,feed_dict={holder2:warped_Bm},
                                fetches = [loss_Bm], loss_callback = self.callback)
                    
                    RA = VGG19(Al_relu41).trueresult(sub_id = order[i])
                    RBm = VGG19(Blm_relu41).trueresult(sub_id = order[i])
                    
                    Al_1 =  self.A_feat[i+1].eval()
                    Al_1  = np.squeeze(Al_1)
                    Al_1_norm  = np.sqrt(np.sum(Al_1**2,axis=(0,1),keepdims = True))
                    Al_1_norm  = (Al_1_norm-np.min(Al_1_norm))/(np.max(Al_1_norm)-np.min(Al_1_norm))
                    #Al_1_norm  = Al_1_norm[:,:,np.newaxis]
                    MA = 1/(1+np.exp(-300*(Al_1_norm**2)-0.05))
                    WA = self.alpha[i]*MA
                    
                    Bl_1 =  self.B_feat[i+1].eval()
                    Bl_1  = np.squeeze(Bl_1)
                    Bl_1_norm  = np.sqrt(np.sum(Bl_1**2,axis=(0,1),keepdims = True))
                    Bl_1_norm  = (Bl_1_norm-np.min(Bl_1_norm))/(np.max(Bl_1_norm)-np.min(Bl_1_norm))
                    #Bl_1_norm  = Bl_1_norm[:,:,np.newaxis]
                    MB = 1/(1+np.exp(-300*(Bl_1_norm**2)-0.05))
                    WB = self.alpha[i]*MB                    
                
                    Al_m = Al_1*WA + np.squeeze(RBm.eval())*(1-WA)
                    Al_m_norm = Al_m/np.sqrt(np.sum(Al_m**2))
                    Bl   = Bl_1*WB + np.squeeze(RA.eval())*(1-WB)
                    Bl_norm = Bl/np.sqrt(np.sum(Bl**2)) 
                    
                #cv2.imwrite()
                
                
            self.nnf_A,self.nnf_B = nnf_A,nnf_B
            
        return nnf_A,nnf_B

    def callback(self,loss_A):
        self.timer.time_use()
        if self._iter%100 ==0:   
            print('----------------------------')
            print('Now is iter {}.'.format(self._iter))
            print('Use time {}.'.format(self.timer.time_formated))
            print('This 100 iteration use time {:.1f}sec.'.format(self.timer.time_used*100))
            print('Expected to finished training in {}.'.format(self.timer.expect_time(self._iter,self.epoches)))
            print('The loss is {:.1f}'.format(loss_A))
            print('----------------------------')
        self._iter += 1    
        
    def rebuildimg(self):
        resA,resB = reconstruct_result(self.A,self.B,self.nnf_A,self.nnf_B,1)
        
        return resA,resB
    @staticmethod
    def sqrloss(dst_value,content_value):
        return tf.reduce_mean(tf.square(dst_value-content_value))      

'''
    import PIL.Image as Image
    map_img_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\Visual attribute\pic\A2.png'                                    
    mapped_img_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\Visual attribute\pic\BP3.png'                                    
                                        
    mapped_img = utils.get_img_PIL(mapped_img_path, max_size = 448)
                                        
    map_img = utils.get_img_PIL(map_img_path, max_size = 448)
    
    A,B = utils.same_shape(mapped_img,map_img)
'''
    
try_ = VisualMapping(B,A)
try_.get_feat()
sess = tf.Session()

nnf_A,nnf_B = try_.NNF_computing(sess)
resA,resB = reconstruct_result(try_.A,try_.B,try_.nnf_A,try_.nnf_B,1)







