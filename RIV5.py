# -*- coding: utf-8 
"""
Created on Fri Mar  8 20:01:41 2024

@author: AKourgli
"""


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

thresholds = np.array([30,100,160,230])
A_reg1 = np.digitize(A_gray, bins=thresholds)

thresholds = threshold_multiotsu(A_gray,3)
A_reg2 = np.digitize(A_gray, bins=thresholds)

plt.figure(1)   
plt.subplot(231); plt.imshow(A_rgb); plt.title('RGB');plt.axis('off')
plt.subplot(232); plt.imshow(A_Sobel_Seuil,cmap='gray'); plt.title('Segmentation contours');plt.axis('off')
plt.subplot(233); plt.plot(h)
plt.subplot(234); plt.imshow(A_reg1,cmap='hsv'); plt.title('Segmentation régions par seuillagage manuel de V');plt.axis('off')
plt.subplot(235); plt.imshow(A_reg2,cmap='hsv'); plt.title('Segmentation régions par seuillage multi-otsu de V');plt.axis('off')

A_HSV = cv2.cvtColor(A_rgb, cv2.COLOR_RGB2HSV); 
plt.subplot(236); plt.imshow(A_HSV[:,:,0]/16,cmap='hsv'); plt.title('Image HSV quantifié')

# Faire binarisataion 
# https://www.kongakura.fr/article/OpenCV_Python_Tutoriel
#th=cv2.bitwise_or(th_h,th_s)



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
# A = cv2.imread('fge.jpeg')
# L,C,D = np.shape(A)
# A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB); A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

AV = A_rgb.reshape((-1,3)); AV = np.float32(AV)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
compac,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center=np.round(center); center = np.uint8(center)
AV_seg = center[label.flatten()]
A_Seg = AV_seg.reshape((A_rgb.shape))

plt.figure(3)
plt.subplot(221); plt.imshow(A_rgb,); plt.title('RGB');plt.axis('off')
plt.subplot(222); plt.imshow(A_Seg);  plt.title('Image segmentée par Kmeans K=6');plt.axis('off')

# A_HSV = cv2.cvtColor(A_rgb, cv2.COLOR_RGB2HSV); A_HSV[:,:,2] = cv2.bilateralFilter(A_HSV[:,:,2], 25, 75, 75); B_rgb = cv2.cvtColor(A_HSV, cv2.COLOR_HSV2RGB)
# AV = B_rgb.reshape((-1,3)); AV = np.float32(AV)
# compac,label,center=cv2.kmeans(AV,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# center=np.round(center); center = np.uint8(center)
# AV_seg = center[label.flatten()]
# A_Seg = AV_seg.reshape((A_rgb.shape))
# plt.subplot(223); plt.imshow(A_Seg); plt.title('Filtée puis segmentée par KMeans')

hr = cv2.calcHist([A_rgb],[0],None,[256],[0,255])
hg = cv2.calcHist([A_rgb],[1],None,[256],[0,255])
hb = cv2.calcHist([A_rgb],[2],None,[256],[0,255])
plt.subplot(223); plt.plot(hb,label='Bleu');plt.plot(hr,label='Rouge');plt.plot(hg,label='Vert');plt.legend()


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



'Mean shift'

from sklearn.cluster import MeanShift, estimate_bandwidth

AV = np.reshape(A_rgb, [-1, 3]);AV = np.float32(AV)

bandwidth = estimate_bandwidth(AV, quantile=.06, n_samples=3000) #It defines the radius of the area in feature space to be considered for computing the mean shift. If not set, it is estimated using a provided heuristic.
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300).fit(AV)
#bin_seeding is a faster way to compute mean shift. If set to True, initial seed points are placed in a grid, reducing the number of iterations needed to converge, which can significantly speed up the algorithm. 

labels = ms.labels_
cluster_centers = ms.cluster_centers_; 
#labels_unique = np.unique(labels);n_clusters_ = len(labels_unique)
AV_seg2 = cluster_centers[labels.flatten()]
A_reg2 = AV_seg2.reshape((A_rgb.shape))
A_reg2=np.uint8(A_reg2)

# plt.figure(4)
# plt.subplot(121);plt.imshow(A_rgb);plt.axis('off')
plt.subplot(224);plt.imshow(A_reg2);plt.axis('off'); plt.title('Segmentée par MeanShift')



# A = cv2.imread('fge.jpeg')
# L,C,D = np.shape(A)
# A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB); A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

# AV = np.reshape(A_rgb, [-1, 3])
# bandwidth = estimate_bandwidth(AV, quantile=.06, n_samples=3000)
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300).fit(AV)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# AV_seg2 = cluster_centers[labels.flatten()]
# A_reg3 = AV_seg2.reshape((A_rgb.shape))
# A_reg3=np.uint8(A_reg3)

# plt.figure(5)
# plt.subplot(221);plt.imshow(A_rgb);plt.axis('off')
# plt.subplot(222);plt.imshow(A_reg3,cmap='viridis');plt.axis('off')

# ms = MeanShift(bandwidth=bandwidth/3, bin_seeding=True, max_iter=300).fit(AV)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# AV_seg2 = cluster_centers[labels.flatten()]
# A_reg3 = AV_seg2.reshape((A_rgb.shape))
# A_reg3=np.uint8(A_reg3)
# plt.subplot(223);plt.imshow(A_reg3,cmap='viridis');plt.axis('off')

# ms = MeanShift(bandwidth=bandwidth*1.5, bin_seeding=True, max_iter=300).fit(AV)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_; 
# AV_seg2 = cluster_centers[labels.flatten()]
# A_reg3 = AV_seg2.reshape((A_rgb.shape))
# A_reg3=np.uint8(A_reg3)
# plt.subplot(224);plt.imshow(A_reg3,cmap='viridis');plt.axis('off')


''' KMenas
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
#cv2.destroyAllWindows()

'https://blog.devgenius.io/practical-guide-to-mean-shift-clustering-5fec0277e44b'