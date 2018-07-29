# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:47:35 2018

@author: Administrator
"""
import numpy as np
from numba import jit

#@jit
def window_distance(A,ax,ay,B,bx,by,win_rad):
    dx0 = min(ax , bx , win_rad)
    dx1 = min(A.shape[1] - ax , B.shape[1] - bx , win_rad+1)
    dy0 = min(ay , by , win_rad)
    dy1 = min(A.shape[0] - ay , B.shape[0] - by , win_rad+1)
    
    distance = (np.sum((A[int(ay-dy0):int(ay+dy1),int(ax-dx0):int(ax+dx1)]-B[int(by-dy0):int(by+dy1),int(bx-dx0):int(bx+dx1)])**2))/ ((dx1+dx0) * (dy1+dy0))
    return distance

#@jit
def initialize_nnf(mapped_shape,map_shape):
    nnf = np.zeros(mapped_shape + (2,))
    nnf[:,:,0] = np.random.randint(map_shape[0],size = mapped_shape)
    nnf[:,:,1] = np.random.randint(map_shape[1],size = mapped_shape)
    
    return nnf

#@jit
def initialize_nnf_value(nnf,A,Am,B,Bm,win_rad):
    assert nnf.shape[0:-1] == A.shape[0:-1],r'Dimension mismatched, may include wrong input array.'
    nnf_value = np.zeros(nnf.shape[0:-1])
    for i in range(nnf.shape[0]):
        for j in range(nnf.shape[1]):
            nnf_value[i,j] = window_distance(A,j,i,B,nnf[i,j][1],nnf[i,j][0],win_rad) + \
                             window_distance(Am,j,i,Bm,nnf[i,j][1],nnf[i,j][0],win_rad)           
    return nnf_value

#@jit                
def propogation(nnf,nnf_value,A,Am,B,Bm,win_rad,iter_time =5):
    for iter_ in range(iter_time):
        print('now is iter{}'.format(iter_))
        for i in range(nnf.shape[0]):
            for j in range(nnf.shape[1]):
                
                loc_y_in_b,loc_x_in_b = nnf[i,j][0],nnf[i,j][1]
                distance = nnf_value[i][j]
                
                for k in reversed(range(4)):
                    d = k**2
                    if i-d>0:
                        ry, rx = nnf[i-d, j][0] + d, nnf[i-d, j][1]
                        if ry < B.shape[0]:
                            val = window_distance(A, j, i, B, rx, ry, win_rad) + \
                                  window_distance(Am, j, i, Bm, rx, ry, win_rad)  
                            
                            if val < distance:
                                loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                
                    if j-d>0:
                        ry, rx = nnf[i, j-d][0], nnf[i, j-d][1] + d
                        if rx < B.shape[1]:
                            val = window_distance(A, j, i, B, rx, ry, win_rad) + \
                                  window_distance(Am, j, i, Bm, rx, ry, win_rad)  
                            if val < distance:
                                loc_y_in_b,loc_x_in_b,distance = ry,rx,val
                                
                    if i+d < A.shape[0]:
                        ry, rx = nnf[i+d, j][0]-d, nnf[i+d, j][1] 
                        if ry > 0:
                            val = window_distance(A, j, i, B, rx, ry, win_rad) + \
                                  window_distance(Am, j, i, Bm, rx, ry, win_rad)  
                            if val < distance:
                                loc_y_in_b,loc_x_in_b,distance = ry,rx,val                        
                                
                    if j+d < A.shape[1]:
                        ry, rx = nnf[i, j+d][0], nnf[i, j+d][1] - d
                        if rx > 0:
                            val = window_distance(A, j, i, B, rx, ry, win_rad) + \
                                  window_distance(Am, j, i, Bm, rx, ry, win_rad)  
                            if val < distance:
                                loc_y_in_b,loc_x_in_b,distance = ry,rx,val                                     
                               
                rand_d = min(B.shape[0]//2, B.shape[1]//2)
                while rand_d > 0:
                    try:
                        
                        xmin = max(loc_x_in_b - rand_d, 0)
                        xmax = min(loc_x_in_b + rand_d, B.shape[1])
                        ymin = max(loc_y_in_b - rand_d, 0)
                        ymax = min(loc_y_in_b + rand_d, B.shape[0])
                    #print(xmin, xmax)
                        rx = np.random.randint(xmin, xmax)
                        ry = np.random.randint(ymin, ymax)
                        val = window_distance(A, j, i, B, rx, ry, win_rad ) + \
                              window_distance(Am, j, i, Bm, rx, ry, win_rad)  
                        if val < distance:
                            loc_y_in_b, loc_x_in_b, distance = ry, rx, val
                    except:
                        print('Dimension wrong')
                    rand_d = rand_d//2
                        
                nnf[i,j][0],nnf[i,j][1] = loc_y_in_b, loc_x_in_b
                nnf_value[i, j] = distance
    return nnf,nnf_value

def nnf_upsample(nnf ,size = None):
    ah, aw = nnf.shape[:2]

    if size is None:
        size = [ah * 2, aw * 2]

    bh, bw = size
    ratio_h, ratio_w = bh / ah, bw / aw
    target = np.zeros(shape=(size[0], size[1], 2)).astype(np.int)

    for by in range(bh):
        for bx in range(bw):
            quot_h, quot_w = int(by // ratio_h), int(bx // ratio_w)
            # print(quot_h, quot_w)
            rem_h, rem_w = (by - quot_h * ratio_h), (bx - quot_w * ratio_w)
            vy, vx = nnf[quot_h, quot_w]
            vy = int(ratio_h * vy + rem_h)
            vx = int(ratio_w * vx + rem_w)
            target[by, bx] = [vy, vx]

    return target

#@jit
def reconstruct(A,B,nnf):
    ans = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            pos = [nnf[i,j][0],nnf[i,j][1]]
            ans[i,j] = B[int(pos[0]), int(pos[1])]
    return ans    

#@jit
def reconstruct_result(A,B,nnf_A,nnf_B,win_rad):
    ans_A = np.zeros_like(A)
    ans_B = np.zeros_like(B)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            pos = [nnf_A[i,j][0],nnf_A[i,j][1]]
            window = B[int(max(pos[0]-win_rad,0)):int(min(pos[0]+win_rad+1,B.shape[0])),
                       int(max(pos[1]-win_rad,0)):int(min(pos[1]+win_rad+1,B.shape[1]))]
            ans_A[i,j] = np.mean(window,axis =(0,1))
            
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            pos = [nnf_B[i,j][0],nnf_B[i,j][1]]
            window = A[int(max(pos[0]-win_rad,0)):int(min(pos[0]+win_rad+1,A.shape[0])),
                       int(max(pos[1]-win_rad,0)):int(min(pos[1]+win_rad+1,A.shape[1]))]
            ans_B[i,j] = np.mean(window,axis = (0,1))        
    return ans_A,ans_B
    
    
    
    