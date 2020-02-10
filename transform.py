# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:18:43 2019

@author: Mohammadmahdi
"""

import numpy as np 
from numpy import linalg as la
 
def pca(img,d):
    m,n,p = np.shape(img)
    sd = np.zeros((p,m*n))
    for i in range(p):
        sd[i,:] = np.reshape(img[:,:,i],(1,m*n))
    M = np.mean(sd,axis=1)
    sd_new = np.transpose(sd) - M
    C = np.cov(np.transpose(sd_new))
    _,v = la.eig(C)
    A = v[:,:d]
    y = np.matmul(sd_new,A)
    Y = np.zeros((m,n,d))
    for i in range(d):
        Y[:,:,i] = np.reshape(y[:,i],(m,n))
        
    return Y