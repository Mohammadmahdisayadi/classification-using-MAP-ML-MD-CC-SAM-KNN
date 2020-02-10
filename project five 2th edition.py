# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:46:43 2019

@author: Mohammad Mahdi Sayadi
"""

import numpy as np 
import math as mt 
import cv2 as cv 
import matplotlib.pyplot as plt
import discriminantfcn as df
import needfcn as nf



sd = np.zeros((80,120,12),dtype="uint8")  # source data storage
edgesd = np.zeros((80,120,12),dtype="uint8")  # source data storage

m,n,p = np.shape(sd)
# p = number of bounds
# m,n = image axis = 0,1 respectively

noc = 8      # number of classes
noC = 8      # number of classifier 
no_tr = 200  # number of trainig samples

gtm = cv.imread('testgtm2.bmp')     # grand truth map
gtm = gtm[:,:,::-1]

plt.figure(1),plt.imshow(gtm)


#%% Reading source image 


for i in range(1,p+1):
    result = str(i)
    list1 = ['testb',result,'.bmp']
    stri = "".join(list1)
    sd[:,:,i-1] = cv.imread(stri,0)
    edgesd[:,:,i-1] = cv.Canny(sd[:,:,i-1].astype("uint8"),90,100)
    plt.figure(2)
    plt.subplot(3,4,i)
    plt.xticks([]),plt.yticks([]),plt.imshow(sd[:,:,i-1],cmap = "gray")
    plt.title(stri)
    plt.figure(3)
    plt.subplot(3,4,i)
    plt.xticks([]),plt.yticks([]),plt.imshow(edgesd[:,:,i-1],cmap = "gray")
    plt.title(stri)
    

sd = sd.astype("float32")
m,n,p = np.shape(sd)

#%%   Peack indices of training samples

ind_storec,ind_storer = nf.patchind(no_tr,noc)

for i in range(noc):
    result = str(i+1)
    list1 = [result,'th class, training samples']
    stri = "".join(list1)
    plt.figure(4)
    plt.subplot(2,4,i+1)
    plt.plot(ind_storec[:,:,i],ind_storer[:,:,i],marker = 'o')
    plt.title(stri)    

 
#%% Fetching training sample 
    
train_sample = np.zeros((p,no_tr,noc),dtype="float32") # each row show correspond class 
 
# In train_sample[a,b,c] : a = shows correspond bound    b = NO. of training samples        c = shows correspond class


for j in range(noc):          # j coresspond to classes
    for i in range(p):       # i correspond to rows
        train_sample[i,:,j] = sd[ind_storer[:,:,j],ind_storec[:,:,j],i]
        


#%% Train MAP classifier  
        
PCi = np.array([[0.25,0.25,1/12,1/12,1/12,1/12,1/12,1/12]]).transpose()     

cov_mat = np.zeros((p,p,noc))
mean_vec = np.zeros((1,p,noc))

for i in range(noc):
    cov_mat[:,:,i] = np.cov(train_sample[:,:,i])
    mean_vec[:,:,i] = np.mean(train_sample[:,:,i],axis = 1)


#%% Evaluate test data for MAP classifier 

gmap = np.zeros((1,noc))
classified_image_MAP = np.zeros((m,n,p))


for i in range(m):
    for j in range(n):
        for k in range(noc):
            gmap[0,k] = df.MAP(np.array([sd[i,j,:]]),mean_vec[:,:,k],cov_mat[:,:,k],PCi[k,0])
            gmap_new = np.argmax(gmap)
            classified_image_MAP[i,j,k] = gmap_new
            classified_image_MAP[i,j,k] = nf.mapper(classified_image_MAP[i,j,k])
                                          
classified_image_MAP = classified_image_MAP.astype("uint8")
plt.figure(5)
plt.imshow(classified_image_MAP[:,:,1],"gray")
plt.title("Classified image using MAP")


#%% Evaluate test data for ML classifier 

gml = np.zeros((1,noc))
classified_image_ML = np.zeros((m,n,p))

for i in range(m):
    for j in range(n):
        for k in range(noc):
            gml[0,k] = df.ML(np.array([sd[i,j,:]]),mean_vec[:,:,k],cov_mat[:,:,k])
            gml_new = np.argmax(gml)
            classified_image_ML[i,j,k] = gml_new
            classified_image_ML[i,j,:] = nf.mapper(classified_image_ML[i,j,k])
            
classified_image_ML = classified_image_ML.astype("uint8")
plt.figure(6)
plt.imshow(classified_image_ML[:,:,7],"gray")
plt.title("Classified image using ML")

#%% Evaluate test data for Minimum Distance classifier 
# Using "One In front of Everyone" approach

classified_image_MD = np.zeros((m,n,p))

for i in range(m):
    for j in range(n):
        for k in range(noc):
            mi = mean_vec[:,:,k]
            if (k==0):
                mj = np.mean(mean_vec[:,:,0:],axis=2)
            elif (k==noc):
                mj = np.mean(mean_vec[:,:,:7],axis=2)
            else:
                mj = (np.mean(mean_vec[:,:,k:],axis=2) + np.mean(mean_vec[:,:,:k],axis=2))/2
            gi = df.MD(np.array([sd[i,j,:]]),mi)
            gj = df.MD(np.array([sd[i,j,:]]),mj)
            gij = gi - gj
            
            if (gij>0):
                classified_image_MD[i,j,:] = k
            classified_image_MD[i,j,:] = nf.mapper(classified_image_MD[i,j,k])
            
classified_image_MD = classified_image_MD.astype("uint8")
plt.figure(7)
plt.imshow(classified_image_MD[:,:,1],"gray")                        
plt.title("Classified image using minimum distance")        
   
#%% Evaluate test data for SAM classifier
# SAM stand for Spectral Angle Mapper 

gsam = np.zeros((1,noc))
classified_image_SAM = np.zeros((m,n,p)) 

for i in range(m):
    for j in range(n):
        for k in range(noc):
            gsam[0,k] = df.SAM(np.array([sd[i,j,:]]),mean_vec[:,:,k])
            gsam_new = np.argmin(gsam)
            classified_image_SAM[i,j,k] = gsam_new
            classified_image_SAM[i,j,:] = nf.mapper(classified_image_SAM[i,j,k])
            
classified_image_SAM = classified_image_SAM.astype("uint8")
plt.figure(8)
plt.imshow(classified_image_SAM[:,:,1],"gray")
plt.title("Classified image using SAM")

#%% Evaluate test data for CC classifier
# CC stand for Correlation Classifier 

gcc = np.zeros((1,noc))
classified_image_CC = np.zeros((m,n,p)) 

for i in range(m):
    for j in range(n):
        for k in range(noc):
            gcc[0,k] = df.CC(np.array([sd[i,j,:]]),mean_vec[:,:,k])
            gcc_new = np.argmax(gmap)
            classified_image_CC[i,j,k] = gcc
            classified_image_CC[i,j,:] = nf.mapper(classified_image_CC[i,j,k])
            
classified_image_CC = classified_image_CC.astype("uint8")
plt.figure(9)
plt.imshow(classified_image_CC[:,:,1],"gray")
plt.title("Classified image using CC") 

#%% Evaluate test data for MF classifier
# MF stand for Match Filter classifier 

gmf = np.zeros((1,noc))
classified_image_MF = np.zeros((m,n,p)) 

for i in range(m):
    for j in range(n):
        for k in range(noc):
            gmf[0,k] = df.MF(np.array([sd[i,j,:]]),mean_vec[:,:,k],cov_mat[:,:,k])
            gmf_new = np.argmax(gmf)
            classified_image_MF[i,j,k] = gmf_new
            classified_image_MF[i,j,:] = nf.mapper(classified_image_MF[i,j,k])
            
classified_image_MF = classified_image_MF.astype("uint8")
plt.figure(10)
plt.imshow(classified_image_MF[:,:,1],"gray")
plt.title("Classified image using match filter")

#%% Evaluate test data for PP classifier
# PP stand for Parallel Pipe classifier

classified_image_PP = np.zeros((m,n,p))

for i in range(m):
    for j in range(n):
            classified_image_PP[i,j,:] = df.PP(sd[i,j,:],train_sample)
            classified_image_PP[i,j,:] = nf.mapper(classified_image_PP[i,j,k])
            
classified_image_PP = classified_image_PP.astype("uint8")
plt.figure()
plt.imshow(classified_image_PP[:,:,1],"gray")
plt.title("Classified image using PP")


#%% Evaluate test data for NN classifier
# Nearest Naibor stand for Parallel Pipe classifier

classified_image_NN = np.zeros((m,n,p))

for i in range(m):
    for j in range(n):
        classified_image_NN[i,j,:] = df.NN(sd[i,j,:],train_sample)

#%% maooing the NN classifier result 


for i in range(m):
    for j in range(n):
        for k in range(noc):
            classified_image_NN[i,j,k] = nf.mapper(classified_image_NN[i,j,k])
            
classified_image_NN = classified_image_NN.astype("uint8")
plt.figure()
plt.imshow(classified_image_NN[:,:,1],"gray")
plt.title("Classified image using NN")                

#%% Writting the result data

classified_image_MAP = classified_image_MAP[:,:,1]
classified_image_ML = classified_image_ML[:,:,1]
classified_image_MD = classified_image_MD[:,:,1]
classified_image_SAM = classified_image_SAM[:,:,1]
classified_image_CC = classified_image_CC[:,:,1]
classified_image_MF = classified_image_MF[:,:,1]  
classified_image_PP = classified_image_PP[:,:,1]
classified_image_NN = classified_image_NN[:,:,1]

#%% confusion matrix of all classifiers 

confusion_matrix = np.zeros((noc+1,noc+1,noC))

interval1,interval2,interval = nf.void()

for k in range(noC):
    for c in range(noc):
        if (c==0):
            for t in range(0,8,2): 
                for i in interval1[t]:
                    for j in interval1[t+1]: 
                        filename = [classified_image_MAP[i,j],classified_image_ML[i,j],
                                    classified_image_MD[i,j],classified_image_SAM[i,j],
                                    classified_image_CC[i,j],classified_image_MF[i,j],
                                    classified_image_PP[i,j],classified_image_NN[i,j]]
                        confusion_matrix[:noc,:noc,k] = nf.counter(filename[k],confusion_matrix[:noc,:noc,k],c)         
        elif (c==1):
            for t in range(0,8,2): 
                for i in interval2[t]:
                    for j in interval2[t+1]: 
                        filename = [classified_image_MAP[i,j],classified_image_ML[i,j],
                                    classified_image_MD[i,j],classified_image_SAM[i,j],
                                    classified_image_CC[i,j],classified_image_MF[i,j],
                                    classified_image_PP[i,j],classified_image_NN[i,j]]
                        confusion_matrix[:noc,:noc,k] = nf.counter(filename[k],confusion_matrix[:noc,:noc,k],c)
        else:
            for i in interval[c-2][0]:
                for j in interval[c-2][1]:
                    filename = [classified_image_MAP[i,j],classified_image_ML[i,j],
                                classified_image_MD[i,j],classified_image_SAM[i,j],
                                classified_image_CC[i,j],classified_image_MF[i,j],
                                classified_image_PP[i,j],classified_image_NN[i,j]]
                    confusion_matrix[:noc,:noc,k] = nf.counter(filename[k],confusion_matrix[:noc,:noc,k],c)
                
#%% OA OV
    
for j in range(noC):    
    for i in range(noc):
        confusion_matrix[i,noc,j] = round(100*(confusion_matrix[i,i,j]/np.sum(confusion_matrix[i,:,j],axis=0)),2)
        confusion_matrix[noc,i,j] = round(100*(confusion_matrix[i,i,j]/np.sum(confusion_matrix[:,i,j],axis=0)),2)
        if (mt.isnan(float(confusion_matrix[noc,i,j]))):
            confusion_matrix[noc,i,j]=0


#%% plotting the data
        
import seaborn as sb 

name1 = ['MAP confusion matrix','ML confusion matrix','MD confusion matrix',
         'SAM confusion matrix','CC confusion matrix','MF confusion matrix',
         'PP confusion matrix','NN confusion matrix']

name2 = ['C1','C2','C3','C4','C5','C6','C7','C8','Acc']  
name3 = ['C1','C2','C3','C4','C5','C6','C7','C8','Val']  
  
for i in range(noC):
    plt.figure(i+1),sb.heatmap(confusion_matrix[:,:,i], xticklabels=name2, yticklabels=name3,annot=True,annot_kws={"size": 10},fmt='.0f')    
    plt.title(name1[i])
       
#%% overall accuracy and average validity 

OA = np.zeros((1,noC))
AV = np.zeros((1,noC))

for i in range(noC):
    OA[0,i] = np.round(100*np.sum(np.diag(confusion_matrix[0:noc,0:noc,i],k=0))/np.sum(np.sum(confusion_matrix[0:noc,0:noc,i])))
    AV[0,i] = np.round((1/noc)*np.sum(confusion_matrix[noc,:,i]))

#%% plotting the OA and AV

# data to plot
n_groups = noC
means_frank = (96, 74, 74, 88, 88, 8,58,97)
means_guido = (95, 64, 64, 84, 84, 2,43,96)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Overall Accuracy')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Average Validity')

plt.xlabel('classifier type')
plt.title('OA and AV')
plt.xticks(index + bar_width, ('MAP', 'ML', 'MD', 'SAM', 'CC', 'MF','PP','NN'))
plt.legend()

nf.autolabel(rects1,ax)
nf.autolabel(rects2,ax)

fig.tight_layout()        
            