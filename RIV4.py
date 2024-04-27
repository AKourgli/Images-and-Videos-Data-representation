# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close('all'); 


B = 9*np.ones((10,10)); B[2:6,3:8]=90; B[8,2]=90; B[4,4]=0; B = np.uint8(B)
k11 = np.array([[1, 0, -1], [1, 0, -1],  [1, 0, -1]])/3; B_k11= cv2.filter2D(src=B, ddepth=5, kernel=k11); 
k12 = np.array([[1, 1, 1], [0, 0, 0],  [-1, -1, -1]])/3; B_k12= cv2.filter2D(src=B, ddepth=5, kernel=k12); 
B_Prewitt = np.sqrt(B_k11**2+B_k12**2);  B_Prewitt = np.round(B_Prewitt); 
k21 = np.array([[1, 0, -1], [2, 0, -2],  [1, 0, -1]])/3; B_k21= cv2.filter2D(src=B, ddepth=5, kernel=k21); 
k22 = np.array([[1, 2, 1], [0, 0, 0],  [-1, -2, -1]])/4; B_k22= cv2.filter2D(src=B, ddepth=5, kernel=k22); 
B_Sobel = np.sqrt(B_k21**2+B_k22**2);  B_Sobel = np.round(B_Sobel); 

k13 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])/4; B_LP1= cv2.filter2D(src=B, ddepth=3, kernel=k13); 
k14 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])/8; B_LP2= cv2.filter2D(src=B, ddepth=3, kernel=k14); 
k15 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])/4; B_LPG= cv2.filter2D(src=B, ddepth=3, kernel=k15);

#Load the image
A = cv2.imread('fge2.jpg')
L,C,D = np.shape(A)
A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB); A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

k11 = np.array([[-1, 0, 1], [-2, 0, 2],  [-1, 0, 1]])/4; A_k11= cv2.filter2D(src=A_gray, ddepth=5, kernel=k11); 
k12 = np.array([[-1, -2, -1], [0, 0, 0],  [1, 2, 1]])/4; A_k12= cv2.filter2D(src=A_gray, ddepth=5, kernel=k12); 
A_Sobel = np.sqrt(A_k11**2+A_k12**2); 

A_k11 = cv2.normalize(A_k11, 0, 255.0, cv2.NORM_MINMAX);A_k11 = np.around(A_k11); A_k11 = np.uint8(A_k11)
A_k12 = cv2.normalize(A_k12, 0, 255.0, cv2.NORM_MINMAX);A_k12 = np.around(A_k12); A_k12 = np.uint8(A_k12); 


plt.figure(3)   
plt.subplot(221); plt.imshow(A_rgb); plt.title('RGB')
plt.subplot(222); plt.imshow(A_Sobel,cmap='gray'); plt.title('Filtre Sobel')
plt.subplot(223); plt.imshow(A_k11,cmap='gray'); plt.title('Gradient Horizontal')
plt.subplot(224); plt.imshow(A_k12,cmap='gray'); plt.title('Gradient Vertical')


A_Sobel_Seuil=np.where(A_Sobel> 50, 255, 0);

k13 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])/4; #k13 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])/8;
A_LP= cv2.filter2D(src=A_gray, ddepth=5, kernel=k13);
A_LP=np.where((A_LP> 30)|(A_LP <-30), 255, 0);

k15 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])/4; A_LPG= cv2.filter2D(src=A_gray, ddepth=5, kernel=k15); 
A_LPG=np.where((A_LP> 30)|(A_LPG <-30), 255, 0);

plt.figure(4)   
plt.subplot(231); plt.imshow(A_gray,cmap='gray'); plt.title('RGB')
plt.subplot(232); plt.imshow(A_Sobel,cmap='gray'); plt.title('Filtre Sobel')
plt.subplot(233); plt.imshow(A_Sobel_Seuil,cmap='gray'); plt.title('Filtre Sobel seuillé')
plt.subplot(234); plt.imshow(A_LP,cmap='gray'); plt.title('Filtre Laplacien')
plt.subplot(235); plt.imshow(A_LPG,cmap='gray'); plt.title('Filtre Laplacien Gaussien')


# ''' Canny : Reduce Noise using Gaussian Smoothing.Compute image gradient using Sobel filter. 
# Apply Non-Max Suppression or NMS to just jeep the local maxima. 
# Finally, apply Hysteresis thresholding which that 2 threshold values T_upper and T_lower'''
A_Canny = cv2.Canny(A_gray, 50 , 200 ) 
plt.subplot(236); plt.imshow(A_Canny,cmap='gray'); plt.title('Filtre Canny')
  


