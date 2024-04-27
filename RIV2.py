# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close('all'); 
 
# Load the image
A = cv2.imread('p4.bmp')
L,C,D = np.shape(A)

A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

# plt.figure(1)
# show 3 RGB bands
plt.subplot(341); plt.imshow(A_rgb[:,:,0],cmap='gray'); plt.title('Red')
plt.subplot(342); plt.imshow(A_rgb[:,:,1],cmap='gray'); plt.title('Green')
plt.subplot(343); plt.imshow(A_rgb[:,:,2],cmap='gray'); plt.title('Blue')
plt.subplot(344); plt.imshow(A_rgb); plt.title('RGB')


''' Histogrammes '''
hh=np.zeros(256)
for i in A_rgb[:,:,2]:
    hh[i]+=1;

for i in range(3):
    h = cv2.calcHist([A_rgb],[i],None,[256],[0,255])
    plt.subplot(3,4,(i+5)), plt.plot(h); 
    plt.subplot(3,4,8), plt.plot(h); 
    h = np.cumsum(h)
    plt.subplot(3,4,(i+9)), plt.plot(h/max(h)); 

''' Ealisation Histogrammes '''  
A_HSV = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
A_HSV[:,:,2] = cv2.equalizeHist(A_HSV[:,:,2]);
A_EQ = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)

plt.figure(2)
# show 3 RGB bands
plt.subplot(341); plt.imshow(A_EQ[:,:,0],cmap='gray'); plt.title('Red')
plt.subplot(342); plt.imshow(A_EQ[:,:,1],cmap='gray'); plt.title('Green')
plt.subplot(343); plt.imshow(A_EQ[:,:,2],cmap='gray'); plt.title('Blue')

plt.subplot(344); plt.imshow(A_rgb); plt.title('RGB')
plt.subplot(3,4,8);plt.imshow(A_HSV[:,:,2],cmap='gray'); plt.title('V égalisé')
plt.subplot(3,4,12);plt.imshow(A_EQ); plt.title('RGB égalisé')


for i in range(3):
    h = cv2.calcHist([A_EQ],[i],None,[256],[0,255])
    plt.subplot(3,4,(i+5)), plt.stem(h); 
    h = np.cumsum(h)
    plt.subplot(3,4,(i+9)), plt.plot(h/max(h)); 

A_rgb_EQ  = np.empty((L,C,D),dtype=np.uint8); 

for i in range(3):
    A_rgb_EQ[:,:,i] = cv2.equalizeHist(A_rgb[:,:,i]);
    
plt.figure(3)   
plt.subplot(221); plt.imshow(A_rgb); plt.title('RGB')
plt.subplot(222); plt.imshow(A_rgb_EQ); plt.title('RGB Egalisé')
plt.subplot(223); plt.imshow(A_rgb); plt.title('RGB')
plt.subplot(224);plt.imshow(A_EQ); plt.title('RGB égalisé par HSV')

