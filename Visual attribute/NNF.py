# -*- coding: utf-8 -*-
"""
Tensorflow Implentation of Style-transfer

@author: FAN
"""
import numpy as np
import PIL.Image as Image
import os


class biNNF:
    def __init__(self,mapped_img, mapped_img_modified, map_img, map_img_modified, win_rad = 1):
        self.A = mapped_img
        self.Am = mapped_img_modified 
        self.B = map_img
        self.Bm = map_img_modified
        self.win_rad = win_rad
        self.nnf_mapped = np.zeros(self.A.shape[0:-1]+(2,))
        self.nnf_map    = np.zeros(self.B.shape[0:-1]+(2,))
        self.nnf_mapped_value = np.zeros(self.A.shape[0:-1])
        self.nnf_map_value    = np.zeros(self.B.shape[0:-1])
        
    def initialize_nnf(self):
        self.nnf_mapped[:,:,0] = np.random.randint(self.B.shape[0],size = self.A.shape[0:-1])
        self.nnf_mapped[:,:,1] = np.random.randint(self.B.shape[1],size = self.A.shape[0:-1])
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                self.nnf_mapped_value[i,j] = self.window_distance(self.A,j,i,self.Bm,self.nnf_mapped[i,j][1],self.nnf_mapped[i,j][0]) + \
                                             self.window_distance(self.Am,j,i,self.B,self.nnf_mapped[i,j][1],self.nnf_mapped[i,j][0])                           
                
        self.nnf_map[:,:,0] = np.random.randint(self.A.shape[0],size = self.B.shape[0:-1])
        self.nnf_map[:,:,1] = np.random.randint(self.A.shape[1],size = self.B.shape[0:-1])
        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):
                self.nnf_map_value[i,j] = self.window_distance(self.B,j,i,self.Am,self.nnf_map[i,j][1],self.nnf_map[i,j][0]) + \
                                             self.window_distance(self.Bm,j,i,self.A,self.nnf_map[i,j][1],self.nnf_map[i,j][0]) 
        
    def window_distance(self,A,ax,ay,B,bx,by):
        dx0 = min(ax , bx , self.win_rad)
        dx1 = min(A.shape[1] - ax , B.shape[1] - bx , self.win_rad+1)
        dy0 = min(ay , by , self.win_rad)
        dy1 = min(A.shape[0] - ay , B.shape[0] - by , self.win_rad+1)
        
        distance = np.sqrt(np.sum((A[int(ay-dy0):int(ay+dy1),int(ax-dx0):int(ax+dx1)]-B[int(by-dy0):int(by+dy1),int(bx-dx0):int(bx+dx1)])**2))/ (dx1+dx0) / (dy1+dy0)
        return distance

    def propogation(self,iter_time = 5):
        
        for iter_ in range(iter_time):
            print('now is iter{}'.format(iter_))
            for i in range(self.A.shape[0]):
                for j in range(self.A.shape[1]):
                    
                    loc_y_in_b,loc_x_in_b = self.nnf_mapped[i,j][0],self.nnf_mapped[i,j][1]
                    distance = self.nnf_mapped_value[i][j]
                    
                    for k in reversed(range(4)):
                        d = k**2
                        if i-d>0:
                            ry, rx = self.nnf_mapped[i-d, j][0] + d, self.nnf_mapped[i-d, j][1]
                            if ry < self.B.shape[0]:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                    
                        if j-d>0:
                            ry, rx = self.nnf_mapped[0][i, j-d], self.nnf_mapped[1][i, j-d] + d
                            if rx < self.B.shape[1]:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                    
                        if i+d < self.A.shape[0]:
                            ry, rx = self.nnf_mapped[i+d, j][0]-d, self.nnf_mapped[i+d, j][1] 
                            if ry > 0:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val                        
                                    
                        if j+d < self.A.shape[1]:
                            ry, rx = self.nnf[i, j+d][0], self.nnf[i, j+d][1] - d
                            if rx > 0:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val                                     
                                   
                    rand_d = min(self.B.shape[0]//2, self.B.shape[1]//2)
                    while rand_d > 0:
                            xmin = max(loc_x_in_b - rand_d, 0)
                            xmax = min(loc_x_in_b + rand_d, self.B.shape[1])
                            ymin = max(loc_y_in_b - rand_d, 0)
                            ymax = min(loc_y_in_b + rand_d, self.B.shape[0])
                        #print(xmin, xmax)
                            rx = np.random.randint(xmin, xmax)
                            ry = np.random.randint(ymin, ymax)
                            val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                  self.window_distance(self.Am, j, i, self.B, rx, ry)  
                            if val < distance:
                                loc_y_in_b, loc_x_in_b, distance = ry, rx, val

                            rand_d = rand_d//2
                            
                    self.nnf_mapped[i,j][0],self.nnf_mapped[i,j][1] = loc_y_in_b, loc_x_in_b
                    self.nnf_mapped_value[i, j] = distance
                    
            for i in range(self.B.shape[0]):
                for j in range(self.B.shape[1]):
                    
                    loc_y_in_b,loc_x_in_b = self.nnf_map[i,j][0],self.nnf_map[i,j][1]
                    distance = self.nnf_map_value[i,j]
                    
                    for k in reversed(range(4)):
                        d = k**2
                        if i-d>0:
                            ry, rx = self.nnf_map[i-d, j][0] + d, self.nnf_map[i-d, j][1]
                            if ry < self.A.shape[0]:
                                val = self.window_distance(self.B, j, i, self.Am, rx, ry) + \
                                      self.window_distance(self.Bm, j, i, self.A, rx, ry)  
                                
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                    
                        if j-d>0:
                            ry, rx = self.nnf_map[0][i, j-d], self.nnf_map[1][i, j-d] + d
                            if rx < self.B.shape[1]:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                    
                        if i+d < self.A.shape[0]:
                            ry, rx = self.nnf[0][i+d, j]-d, self.nnf[1][i+d, j] 
                            if ry > 0:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val                        
                                    
                        if j+d < self.A.shape[1]:
                            ry, rx = self.nnf[0][i, j+d], self.nnf[1][i, j+d] - d
                            if rx > 0:
                                val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                      self.window_distance(self.Am, j, i, self.B, rx, ry)  
                                if val < distance:
                                    loc_y_in_b,loc_x_in_b,distance = ry,rx,val                                     
                                   
                    rand_d = min(self.B.shape[0]//2, self.B.shape[1]//2)
                    while rand_d > 0:
                            xmin = max(loc_x_in_b - rand_d, 0)
                            xmax = min(loc_x_in_b + rand_d, self.B.shape[1])
                            ymin = max(loc_y_in_b - rand_d, 0)
                            ymax = min(loc_y_in_b + rand_d, self.B.shape[0])
                        #print(xmin, xmax)
                            rx = np.random.randint(xmin, xmax)
                            ry = np.random.randint(ymin, ymax)
                            val = self.window_distance(self.A, j, i, self.Bm, rx, ry) + \
                                  self.window_distance(self.Am, j, i, self.B, rx, ry)  
                            if val < distance:
                                loc_y_in_b, loc_x_in_b, distance = ry, rx, val

                            rand_d = rand_d//2
                            
                    self.nnf_mapped[i,j][0],self.nnf_mapped[i,j][1] = loc_y_in_b, loc_x_in_b
                    self.nnf_mapped_value[i, j] = distance                    
        return self.nnf
                                    
    def reconstruct(self):
        ans = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = [self.nnf[0][i][j],self.nnf[1][i][j]]
                ans[i,j] = self.B[int(pos[0]), int(pos[1])]
        return ans                                    


def window_distance(A,ax,ay,B,bx,by,win_rad):
    dx0 = min(ax , bx , win_rad)
    dx1 = min(A.shape[1] - ax , B.shape[1] - bx , win_rad+1)
    dy0 = min(ay , by , win_rad)
    dy1 = min(A.shape[0] - ay , B.shape[0] - by , win_rad+1)
    
    distance = np.sqrt(np.sum((A[int(ay-dy0):int(ay+dy1),int(ax-dx0):int(ax+dx1)]-B[int(by-dy0):int(by+dy1),int(bx-dx0):int(bx+dx1)])**2))/ (dx1+dx0) / (dy1+dy0)
    return distance


def initialize_nnf(mapped_shape,map_shape):
    nnf = np.zeros(mapped_shape + (2,))
    nnf[:,:,0] = np.random.randint(map_shape[0],size = mapped_shape)
    nnf[:,:,1] = np.random.randint(map_shape[1],size = mapped_shape)
    
    return nnf

def initialize_nnf_value(nnf,A,Am,B,Bm):
    assert nnf.shape[0:-1] == A.shape[0:-1],r'Dimension mismatched, may include wrong input array.'
    nnf_value = np.zeros(nnf.shape[0:-1])
    for i in range(nnf.shape[0]):
        for j in range(nnf.shape[1]):
            nnf_value[i,j] = window_distance(A,j,i,Bm,nnf[i,j][1],nnf[i,j][0]) + \
                             window_distance(Am,j,i,B,nnf[i,j][1],nnf[i,j][0])           
    return nnf_value
                
















                                    

                                    
map_img_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\Neural artistic style\content_src\forest.jpg'                                    
mapped_img_path = r'E:\python.file\git-Repository\TF--style-transfer-collection\Neural artistic style\dst\content1_2-3_2\style-transfer_2001.jpg'                                    
                                    
map_img = Image.open(map_img_path)
mapped_img = Image.open(mapped_img_path)                                    
                                    
map_img = np.float32(map_img)
mapped_img = np.float32(mapped_img)                                    
                                    
                        
nnf = NNF(mapped_img=mapped_img,map_img=map_img)
nnf.initialize_nnf()
origin = nnf.nnf
nnf_matrix = nnf.train()

img_ans = np.uint8(nnf.reconstruct())
