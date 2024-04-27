# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close('all'); 


B = 9*np.ones((10,10)); B[2:6,3:8]=90; B[8,2]=90; B[4,4]=0; B = np.uint8(B)
k1 = np.array([[0, 0, 0], [0, 1, 0],  [0, 0, 0]]); B_k1= cv2.filter2D(src=B, ddepth=-1, kernel=k1);
k2 = np.array([[1, 1, 1], [1, 1, 1],  [1, 1, 1]])/9; B_k2= cv2.filter2D(src=B, ddepth=-1, kernel=k2, borderType=cv2.BORDER_REFLECT101); B_k2 = np.around(B_k2); B_k2 = np.uint8(B_k2)
#k2 = np.ones((5,5))/25; B_k2= cv2.filter2D(src=B, ddepth=-1, kernel=k2); B_k2 = np.around(B_k2); B_k2 = np.uint8(A_k2)
k3 = np.array([[1, 2, 1], [2, 4, 2],  [1, 2, 1]])/16; B_k3= cv2.filter2D(src=B, ddepth=-1, kernel=k3);B_k3 = np.around(B_k3); B_k3 = np.uint8(B_k3)
k4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]); B_k4= cv2.filter2D(src=B, ddepth=-1, kernel=k4); 
k5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])/5; B_k5= cv2.filter2D(src=B, ddepth=5, kernel=k5); 
B_k6 = cv2.medianBlur(B, 3)
k11 = np.array([[1, 0, -1], [1, 0, -1],  [1, 0, -1]])/3; B_k11= cv2.filter2D(src=B, ddepth=5, kernel=k11); 
k12 = np.array([[1, 1, 1], [0, 0, 0],  [-1, -1, -1]])/3; B_k12= cv2.filter2D(src=B, ddepth=5, kernel=k12); 
B_Prewitt = np.sqrt(B_k11**2+B_k12**2); 
#B_Prewitt = cv2.normalize(B_Prewitt, 0, 255.0, cv2.NORM_MINMAX) ; B_Prewitt = np.around(B_Prewitt); B_Prewitt = np.uint8(B_Prewitt)
#k13 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]); B_LP= cv2.filter2D(src=B, ddepth=5, kernel=k13);
#B_LP = cv2.normalize(B_Prewitt, 0, 255.0, cv2.NORM_MINMAX) ; B_Prewitt = np.around(B_Prewitt); B_Prewitt = np.uint8(B_Prewitt)

 
#Load the image
#A = cv2.imread('p4.bmp')
A = cv2.imread('fge2.jpg')
L,C,D = np.shape(A)
A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)


''' Filtrage '''
mean = 0 ; stddev = 81; noise = np.zeros((L,C), np.uint8); cv2.randn(noise, mean, stddev); 
A_HSV = cv2.cvtColor(A_rgb, cv2.COLOR_RGB2HSV); A_HSV[:,:,2] = cv2.add(A_HSV[:,:,2], noise); B_rgb = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)

k2 = np.array([[1, 1, 1], [1, 1, 1],  [1, 1, 1]])/9; A_k2= cv2.filter2D(src=B_rgb, ddepth=-1, kernel=k2); A_k2 = np.around(A_k2); A_k2 = np.uint8(A_k2)
#k2 = np.ones((5,5))/25; A_k2= cv2.filter2D(src=A_rgb, ddepth=-1, kernel=k2); A_k2 = np.around(A_k2); A_k2 = np.uint8(A_k2)
k3 = np.array([[1, 2, 1], [2, 4, 2],  [1, 2, 1]])/16; A_k3= cv2.filter2D(src=B_rgb, ddepth=-1, kernel=k3); A_k3 = np.around(A_k3);A_k3 = np.uint8(A_k3)
k4 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]); A_k4= cv2.filter2D(src=B_rgb, ddepth=-1, kernel=k4); A_k4 = np.around(A_k4);A_k4 = np.uint8(A_k4)

plt.figure(1)   
plt.subplot(221); plt.imshow(B_rgb); plt.title('RGB')
plt.subplot(222); plt.imshow(A_k2); plt.title('Filtre moyenneur 3x3')
plt.subplot(223); plt.imshow(A_k3); plt.title('Filtre Gaussien 3x3')
plt.subplot(224); plt.imshow(A_k4); plt.title('Filtre de Rehaussement')


A_HSV = cv2.cvtColor(B_rgb, cv2.COLOR_RGB2HSV); A_HSV[:,:,2] = cv2.medianBlur(A_HSV[:,:,2], 3); A_k4 = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)
A_HSV = cv2.cvtColor(B_rgb, cv2.COLOR_RGB2HSV); A_HSV[:,:,2] = cv2.bilateralFilter(A_HSV[:,:,2], 15, 75, 75); A_k5 = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)

plt.figure(2)   
plt.subplot(231); plt.imshow(A_rgb); plt.title('RGB')
plt.subplot(232); plt.imshow(B_rgb); plt.title('RGB bruité')
plt.subplot(233); plt.imshow(A_k2); plt.title('Filtre moyenneur 3x3')
plt.subplot(234); plt.imshow(A_k3); plt.title('Filtre Gaussien 3x3')
plt.subplot(235); plt.imshow(A_k4); plt.title('Filtre Median')
plt.subplot(236); plt.imshow(A_k5); plt.title('Filtre Bilateral')

