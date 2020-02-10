# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:40:42 2019

@author: Mohammadmahdi
"""
import numpy as np 



def mapper(inp):
    if (inp==0):
        inp = 32
    elif (inp==1):
        inp = 64
    elif (inp==2):
        inp = 96
    elif (inp==3):
        inp = 128
    elif (inp==4):
        inp = 159
    elif (inp==5):
        inp = 191
    elif (inp==6):
        inp = 223
    elif (inp==7):
        inp = 0  
    return inp

def counter(condition,confusion,c):
    if (condition==32):
        confusion[c,0] = confusion[c,0] + 1
    elif (condition==64):
        confusion[c,1] = confusion[c,1] + 1
    elif (condition==96):
        confusion[c,2] = confusion[c,2] + 1
    elif (condition==128):
        confusion[c,3] = confusion[c,3] + 1
    elif (condition==159):
        confusion[c,4] = confusion[c,4] + 1
    elif (condition==191):
        confusion[c,5] = confusion[c,5] + 1
    elif (condition==223):
        confusion[c,6] = confusion[c,6] + 1
    elif (condition==0):
        confusion[c,7] = confusion[c,7] + 1
    return confusion 


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom') 

def void():
    ####################################
    a1r = [i for i in range(20,40)]
    a1c = [i for i in range(20,60)]
    b1r = [i for i in range(40,60)]
    b1c = [i for i in range(60,80)]
    c1r = [i for i in range(40)]
    c1c = [i for i in range(100,120)]
    d1r = [i for i in range(40,60)]
    d1c = [i for i in range(20)]
    ####################################
    a2r = [i for i in range(40,60)]
    a2c = [i for i in range(20,60)]
    b2r = [i for i in range(20,40)]
    b2c = [i for i in range(60,80)]
    c2r = [i for i in range(20,40)]
    c2c = [i for i in range(20)]
    d2r = [i for i in range(40,80)]
    d2c = [i for i in range(100,120)]
    ####################################
    a3r = [i for i in range(40,80)]
    a3c = [i for i in range(80,100)]
    ####################################
    a4r = [i for i in range(20)]
    a4c = [i for i in range(40,80)]
    ####################################
    a5r = [i for i in range(20)]
    a5c = [i for i in range(40)]
    ####################################
    a6r = [i for i in range(40)]
    a6c = [i for i in range(80,100)]
    ####################################
    a7r = [i for i in range(60,80)]
    a7c = [i for i in range(40,80)]
    ####################################
    a8r = [i for i in range(60,80)]
    a8c = [i for i in range(40)]
    ####################################
    interval1 = [a1r,a1c,b1r,b1c,c1r,c1c,d1r,d1c]
    interval2 = [a2r,a2c,b2r,b2c,c2r,c2c,d2r,d2c]
    interval = [[a3r,a3c],[a4r,a4c],[a5r,a5c],[a6r,a6c],[a7r,a7c],[a8r,a8c]]
    
    return (interval1,interval2,interval)

def patchind(no_tr,noc):
    ind_storer = np.zeros((1,no_tr,noc),dtype="int")
    ind_storec = np.zeros((1,no_tr,noc),dtype="int")
    
    ######################################## 1th variable indices    
        
    a1r = np.random.randint(20,39, size=(1, 50))  
    a1c = np.random.randint(20,59, size=(1, 50))
    b1r = np.random.randint(40,59, size=(1, 50))  
    b1c = np.random.randint(60,79, size=(1, 50))
    c1r = np.random.randint(39, size=(1, 50))  
    c1c = np.random.randint(100,119, size=(1, 50))
    d1r = np.random.randint(40,59, size=(1, 50))  
    d1c = np.random.randint(19, size=(1, 50))
    
    ind_storec[:,:,0] = np.concatenate((a1c,b1c,c1c,d1c),axis=1) 
    ind_storer[:,:,0] = np.concatenate((a1r,b1r,c1r,d1r),axis=1)
    
    ######################################## 2th variable indices 
    
    a2r = np.random.randint(40,59, size=(1, 50))  
    a2c = np.random.randint(20,59, size=(1, 50))
    b2r = np.random.randint(20,39, size=(1, 50))  
    b2c = np.random.randint(60,79, size=(1, 50))
    c2r = np.random.randint(20,39, size=(1, 50))  
    c2c = np.random.randint(19, size=(1, 50))
    d2r = np.random.randint(40,79, size=(1, 50))  
    d2c = np.random.randint(100,119, size=(1, 50))
    
    ind_storec[:,:,1] = np.concatenate((a2c,b2c,c2c,d2c),axis=1) 
    ind_storer[:,:,1] = np.concatenate((a2r,b2r,c2r,d2r),axis=1)
    
    ######################################## 3th variable indices
    
    ind_storec[:,:,2] = np.random.randint(80,99, size=(1, 200))
    ind_storer[:,:,2] = np.random.randint(40,79, size=(1, 200))
    
    ######################################## 4th variable indices
      
    ind_storec[:,:,3] = np.random.randint(40,79, size=(1, 200))
    ind_storer[:,:,3] = np.random.randint(19, size=(1, 200))
    
    ######################################## 5th variable indices
    
    ind_storec[:,:,4] = np.random.randint(39, size=(1, 200))
    ind_storer[:,:,4] = np.random.randint(19, size=(1, 200))
    
    ######################################## 6th variable indices
    
    ind_storec[:,:,5] = np.random.randint(80,99, size=(1, 200))
    ind_storer[:,:,5] = np.random.randint(39, size=(1, 200))
    
    ######################################## 7th variable indices
    
    ind_storec[:,:,6] = np.random.randint(40,79, size=(1, 200))
    ind_storer[:,:,6] = np.random.randint(60,79, size=(1, 200))
    
    ######################################## 8th variable indices
    
    ind_storec[:,:,7] = np.random.randint(39, size=(1, 200))
    ind_storer[:,:,7] = np.random.randint(60,79, size=(1, 200))
    
    return (ind_storec,ind_storer)
    
                             