# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:14:30 2024

@author: AKourgli
"""

import pywt
import matplotlib.pyplot as plt
import numpy as np; import cv2
from copy import deepcopy

# Load the image
A_bgr = plt.imread('fge2.jpg')
#image=A_bgr
image = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
plt.close('all')

wavelet = 'haar' 
coeffs = pywt.wavedec2(image, wavelet, level=2) 
cf=deepcopy(coeffs)
[cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]=cf

cv2.normalize(cA2,cA2,0,255,cv2.NORM_MINMAX);cv2.normalize(cH2,cH2,0,255,cv2.NORM_MINMAX);
cv2.normalize(cV2,cV2,0,255,cv2.NORM_MINMAX);cv2.normalize(cD2,cD2,0,255,cv2.NORM_MINMAX);
cv2.normalize(cH1,cH1,0,255,cv2.NORM_MINMAX);cv2.normalize(cV1,cV1,0,255,cv2.NORM_MINMAX);
cv2.normalize(cD1,cD1,0,255,cv2.NORM_MINMAX);

L,C = np.shape(cA2); level2 = np.empty((2*L,2*C)); 
level2[:L,:C]=cA2;level2[L:2*L,:C]=cV2;level2[:L,C:2*C]=cH2; level2[L:2*L,C:2*C]=cD2;
L1,C1 = np.shape(cH1);level1 = np.empty((2*L1,2*C1));
level1[:L1,:C1]=level2[:L1,:C1];level1[L1:2*L1,:C1]=cV1;level1[:L1,C1:2*C1]=cH1; level1[L1:2*L1,C1:2*C1]=cD1;
cv2.rectangle(level1, (0,0), (0+C1,0+L1), 255,2)

plt.figure(1)
plt.subplot(121);plt.imshow(image, cmap='gray');plt.axis('off')
plt.subplot(122);plt.imshow(level1, cmap='gray');plt.axis('off');plt.title('Decomposition level 2')

A_bgr_l2= np.empty((L,C,3));
for i in range (3):
    cff = pywt.wavedec2(A_bgr[:,:,i], wavelet, level=2) 
    [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]=cff
    A_bgr_l2[:L,:C,i]=cA2[:L,:C]
    
cv2.normalize(A_bgr_l2,A_bgr_l2,0,255,cv2.NORM_MINMAX);
A_bgr_l2 = np.round(A_bgr_l2); A_bgr_l2 = np.uint8(A_bgr_l2);

plt.figure(2)
plt.subplot(121);plt.imshow(A_bgr);plt.axis('off')
plt.subplot(122);plt.imshow(A_bgr_l2);plt.axis('off');plt.title('Approximation level 2')
cv2.imwrite('fge_compressed.jpg', A_bgr_l2);
cv2.imwrite('fge_copy.jpg', A_bgr)

#%% Compression by applying a threshold to reduce detail coefficients
coef_thr = pywt.wavedec2(image, wavelet, level=2) 

threshold = 0.4
for i in range(1, len(coef_thr)):
    coef_thr[i] = list(np.where(np.abs(coef_thr[i]) < threshold*np.max(coef_thr[i]), 0.0, coef_thr[i]));

reconstructed_img = pywt.waverec2(coef_thr, wavelet)

# cv2.normalize(reconstructed_img ,reconstructed_img ,0,255,cv2.NORM_MINMAX);
# reconstructed_img = np.round(reconstructed_img ); reconstructed_img  = np.uint8(reconstructed_img );
cffff=deepcopy(coef_thr)
[cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]=cffff
# cv2.normalize(cA2,cA2,0,255,cv2.NORM_MINMAX);cv2.normalize(cH2,cH2,0,255,cv2.NORM_MINMAX);
# cv2.normalize(cV2,cV2,0,255,cv2.NORM_MINMAX);cv2.normalize(cD2,cD2,0,255,cv2.NORM_MINMAX);
# cv2.normalize(cH1,cH1,0,255,cv2.NORM_MINMAX);cv2.normalize(cV1,cV1,0,255,cv2.NORM_MINMAX);
# cv2.normalize(cD1,cD1,0,255,cv2.NORM_MINMAX);

L,C = np.shape(cA2); level2 = np.empty((2*L,2*C)); 
level2[:L,:C]=cA2;level2[L:2*L,:C]=cV2;level2[:L,C:2*C]=cH2; level2[L:2*L,C:2*C]=cD2;
L,C = np.shape(cH1);level1 = np.empty((2*L,2*C));
level1[:L,:C]=level2[:L,:C];level1[L:2*L,:C]=cV1;level1[:L,C:2*C]=cH1; level1[L:2*L,C:2*C]=cD1;
cv2.rectangle(level1, (0,0), (0+C,0+L), 255,2)

L,C = np.shape(image)
Rate =np.count_nonzero(level1 != 0)/(L*C)*100
print(f"% of coeffcients used for reconstruction: {Rate:.2f}") 

plt.figure(3)
plt.subplot(121);plt.imshow(image, cmap='gray');plt.axis('off')
plt.subplot(122);plt.imshow(level1, cmap='gray');plt.axis('off');plt.title('Decomposition level 2')

plt.figure(4)
plt.subplot(121);plt.imshow(image, cmap='gray');plt.axis('off')
plt.subplot(122);plt.imshow(reconstructed_img, cmap='gray'); plt.title('Compressed Image');plt.axis('off')


#%% Denoising using wavelet
L,C = np.shape(image);
A_HSV = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2HSV); B_HSV= A_HSV;
mean = 0 ; stddev = 25; noise = np.zeros((L,C), np.uint8); cv2.randn(noise, mean, stddev); 
B_HSV[:,:,2] = cv2.add(B_HSV[:,:,2], noise); B_bgr = cv2.cvtColor(B_HSV, cv2.COLOR_HSV2BGR)

plt.figure(5)
plt.subplot(121);plt.imshow(B_bgr);plt.axis('off')

wavelt = 'bior'
cff = pywt.wavedec2(B_HSV[:,:,2], wavelet, level=2) 
threshold = 0.4
for i in range(1, len(cff)):
    cff[i] = list(np.where(np.abs(cff[i]) < threshold*np.max(cff[i])  , 0.0, cff[i]));

B_HSV[:,:,2] = pywt.waverec2(cff, wavelet)
B_bgr = cv2.cvtColor(B_HSV, cv2.COLOR_HSV2BGR)
plt.subplot(122);plt.imshow(B_bgr); plt.title('Denoised Image');plt.axis('off')


