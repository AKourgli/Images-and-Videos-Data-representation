# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_multiotsu


plt.close('all'); 


#Load the image
A = cv2.imread('fge2.jpg')
L,C,D = np.shape(A)
A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB); A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

k11 = np.array([[-1, 0, 1], [-2, 0, 2],  [-1, 0, 1]])/4; A_k11= cv2.filter2D(src=A_gray, ddepth=5, kernel=k11); 
k12 = np.array([[-1, -2, -1], [0, 0, 0],  [1, 2, 1]])/4; A_k12= cv2.filter2D(src=A_gray, ddepth=5, kernel=k12); 
A_Sobel = np.sqrt(A_k11**2+A_k12**2); 
A_Sobel_Seuil=np.where(A_Sobel> 50, 255, 0);


h = cv2.calcHist([A_gray],[0],None,[256],[0,255])
#thresholds = threshold_multiotsu(A_rgb)
thresholds = np.array([0,30,100,160,210,255])
A_reg1 = np.digitize(A_gray, bins=thresholds,right=True)

plt.figure(1)   
plt.subplot(231); plt.imshow(A_gray,cmap='gray'); plt.title('RGB')
plt.subplot(232); plt.imshow(A_Sobel_Seuil,cmap='gray'); plt.title('Segmentation contours')
plt.subplot(233); plt.plot(h)
plt.subplot(234); plt.imshow(A_reg1,cmap='hsv'); plt.title('Segmentation régions par quantification')

A_HSV = cv2.cvtColor(A_rgb, cv2.COLOR_RGB2HSV); 
plt.subplot(235); plt.imshow(A_HSV[:,:,0],cmap='hsv'); plt.title('Image HSV')


# from matplotlib import colors
# B = A_rgb
# r, g, b = cv2.split(B)
# fig = plt.figure(2)
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# pixel_colors = B.reshape((np.shape(B)[0]*np.shape(B)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red");axis.set_ylabel("Green"); axis.set_zlabel("Blue")
# plt.show()

'K means'
AV = A_rgb.reshape((-1,3)); AV = np.float32(AV)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
compac,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center=np.round(center); center = np.uint8(center)
AV_seg = center[label.flatten()]
A_Seg = AV_seg.reshape((A_rgb.shape))

plt.figure(3)
plt.subplot(221); plt.imshow(A_rgb,); plt.title('RGB')
plt.subplot(222); plt.imshow(A_Seg);  plt.title('Image segmentée par Kmeans K=8')

# A_HSV = cv2.cvtColor(A_rgb, cv2.COLOR_RGB2HSV); A_HSV[:,:,2] = cv2.bilateralFilter(A_HSV[:,:,2], 25, 75, 75); B_rgb = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)
# AV = B_rgb.reshape((-1,3)); AV = np.float32(AV)
# compac,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# center=np.round(center); center = np.uint8(center)
# AV_seg = center[label.flatten()]
# A_Seg = AV_seg.reshape((A_rgb.shape))
# plt.subplot(223); plt.imshow(A_Seg); plt.title('Filtée puis segmentée')


# K = 4
# ret,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# center=np.round(center); center = np.uint8(center)
# res = center[label.flatten()]
# A_Seg4 = res.reshape((A_rgb.shape))
# K = 2
# ret,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# center=np.round(center); center = np.uint8(center)
# res = center[label.flatten()]
# A_Seg2 = res.reshape((A_rgb.shape))
# plt.subplot(223); plt.imshow(A_Seg2);  plt.title('Image segmentée par Kmeans K=2')
# plt.subplot(224); plt.imshow(A_Seg4);  plt.title('Image segmentée par Kmeans K=4')


'''
Input parameters
samples : It should be of np.float32 data type, and each feature should be put in a single column.
nclusters(K) : Number of clusters required at end
criteria : It is the iteration termination criteria. When this criteria is satisfied, algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:
type of termination criteria. It has 3 flags as below:
cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
max_iter - An integer specifying maximum number of iterations.
epsilon - Required accuracy
attempts : Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
flags : This flag is used to specify how initial centers are taken. Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.
Output parameters

compactness : It is the sum of squared distance from each point to their corresponding centers.
labels : This is the label array (same as 'code' in previous article) where each element marked '0', '1'.....
centers : This is array of centers of clusters.
'''
# A_reg1 = 255*np.ones((L,C), np.uint8);
# A_reg1=np.where(A_gray<= 30, 0, A_reg1);
# A_reg1=np.where((A_gray>30)& (A_gray<=100), 60, A_reg1);
# A_reg1=np.where((A_gray>100)& (A_gray<=160), 120, A_reg1);
# A_reg1=np.where((A_gray>160)& (A_gray<=210), 180, A_reg1);
# A_reg1=np.where((A_gray>210)& (A_gray<250), 240, A_reg1);
