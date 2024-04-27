# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:43:20 2024

@author: AKourgli
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature


#%% Load image
image_name = 'fge2.jpg'
A_bgr = cv2.imread(image_name)
A_rgb = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB);
image = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
plt.close('all'); 
#%% First Order Statistics ang GLCM Features extraction of ROI chosen manually 
T = 80
ROI = np.empty((7,T,T),np.uint8); mask = np.ones((T,T))
x,y=1100,410 ; ROI[0,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb, (x,y), (x+T,y+T), 255,4)
x,y=1000,1400 ; ROI[1,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 255,4)
x,y=800,1000 ; ROI[2,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 255,4)
x,y=1550,600 ; ROI[3,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 255,4)
x,y=250,1400 ; ROI[4,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 255,4)
x,y=250,600 ; ROI[5,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 255,4)
x,y=0,800 ; ROI[6,:,:] = image[y:y+T,x:x+T]; A_rgb_ROI = cv2.rectangle(A_rgb_ROI, (x,y), (x+T,y+T), 0,4)
plt.figure(1)
plt.imshow(A_rgb_ROI);plt.axis('off')


K = 7
Prop = np.empty((K,6)); 
for i in range(K) :
    ROI_image = ROI[i,:,:]
    glcm = feature.graycomatrix(ROI_image, distances=[1], angles=[0], levels=256, symmetric=True)
    # prop {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
    Prop[i,0] = feature.graycoprops(glcm, 'contrast')
    Prop[i,1] = feature.graycoprops(glcm, 'dissimilarity')
    Prop[i,2] = feature.graycoprops(glcm, 'homogeneity')
    Prop[i,3] = feature.graycoprops(glcm, 'energy')
    Prop[i,4] = feature.graycoprops(glcm, 'correlation')
    Prop[i,5] = feature.graycoprops(glcm, 'ASM')
    
# Features plot
plt.figure(2)
Labels = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM'] 
for i in range(K) :
    plt.subplot(2,7,i+1);plt.imshow(ROI[i,:,:], cmap='gray');plt.title('ROI %d'%i); plt.axis('off')
for i in np.arange(6) :
    plt.subplot(2,6,7+i);plt.stem(Prop[:,i]);plt.title('%s'%Labels[i])

#%% K means Segementation
AV = A_rgb.reshape((-1,3)); AV = np.float32(AV)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
compac,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center=np.round(center); center = np.uint8(center)
AV_seg = center[label.flatten()]
A_Seg = label.reshape((image.shape))
plt.figure(3)
plt.subplot(321);plt.imshow(A_rgb);plt.title('Image originale');plt.axis('off')
plt.subplot(322);plt.imshow(A_Seg, cmap='hsv');plt.axis('off')

#%% GLCM Features extraction of ROI by masking

Prop = np.empty((K,6)); 
for i in range(K) :
    #Create mask for region i
    mask = np.where(A_Seg == i, 1, 0);
    #plt.subplot(3,3,(i+2));plt.imshow(mask, cmap='gray'); plt.title('Initial Mask')
    #Fill image with zeros outside ROI
    ROI_image = np.multiply(image.astype(np.double), mask).astype(np.uint8)
    plt.subplot(3,6,(i+7));plt.imshow(ROI_image, cmap='gray');plt.title('ROI');plt.axis('off')
    ROI_image = ROI[i,:,:]
    glcm = feature.graycomatrix(ROI_image, distances=[1], angles=[0], levels=256, symmetric=True)
    # prop {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
    Prop[i,0] = feature.graycoprops(glcm, 'contrast')
    Prop[i,1] = feature.graycoprops(glcm, 'dissimilarity')
    Prop[i,2] = feature.graycoprops(glcm, 'homogeneity')
    Prop[i,3] = feature.graycoprops(glcm, 'energy')
    Prop[i,4] = feature.graycoprops(glcm, 'correlation')
    Prop[i,5] = feature.graycoprops(glcm, 'ASM')
    
for i in np.arange(6) :
    plt.subplot(3,6,13+i);plt.stem(Prop[:,i]);plt.title('%s'%Labels[i])

#%% LPB Features

radius = 1; n_points = 8 * radius
ker = np.ones((5,5))/25
image = cv2.filter2D(src=image, ddepth=-1, kernel=ker);
lbp8 = feature.local_binary_pattern(image, n_points,radius, method='default'); lbp8 =np.uint8 (lbp8)
hist8, _ = np.histogram(lbp8.ravel(), bins=256, range=(0, 256)); 
lbp8RI = feature.local_binary_pattern(image, n_points,radius, method='ror'); lbp8 =np.uint8 (lbp8)
hist8RI, _ = np.histogram(lbp8RI.ravel(), bins=256, range=(0, 256)); 
lbp8RIu = feature.local_binary_pattern(image, n_points,radius, method='uniform')
hist8RIu, _ = np.histogram(lbp8RIu.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2));hist8RIu = hist8RIu.astype("float"); hist8RIu /= hist8RIu.sum()
plt.figure(4)
plt.subplot(231);plt.imshow(image,cmap='gray');plt.title('Image'); plt.axis('off')
plt.subplot(232);plt.imshow(lbp8,cmap='gray');plt.title('LBP'); plt.axis('off')
plt.subplot(233);plt.imshow(lbp8RI,cmap='gray');plt.title('LBP Rotation Invariant'); plt.axis('off')
plt.subplot(234);plt.imshow(lbp8RIu,cmap='gray');plt.title('LBP uniform'); plt.axis('off')
plt.subplot(235);plt.stem(hist8); plt.title('LBP Histogram'); 
plt.subplot(236);plt.stem(hist8RIu); plt.title('LBP Rotation Invariant Uniform Histogram'); 


plt.figure(5)
F_LBP = np.empty((7,10));
for i in range(K) :
    ROI_image = ROI[i,:,:]
    lbp8RIu = feature.local_binary_pattern(ROI_image, n_points,radius, method='uniform')
    F_LBP[i,:], _ = np.histogram(lbp8RIu.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2)); F_LBP[i,:] = F_LBP[i,:].astype("float"); F_LBP[i,:] /= F_LBP[i,:].sum()
    plt.subplot(4,4,2*i+1);plt.imshow(ROI_image, cmap='gray');plt.axis('off')
    plt.subplot(4,4,2*i+2);plt.stem(F_LBP[i,:])
    
#%%

