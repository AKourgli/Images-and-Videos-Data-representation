# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:00:14 2024

@author: AKourgli
"""

import numpy as np
from skimage.feature import hog
from skimage import exposure
import cv2
import matplotlib.pyplot as plt
plt.close('all')

# # # %% HOG descriptor
# #image_name = 'fge2.jpg'
# image_name = 'fge3.jpg'
# #image_name = 'p4.bmp'
# A_bgr = cv2.imread(image_name)
# A_rgb = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB)
# image = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
# fv, hog_image = hog(image, orientations=9, pixels_per_cell=(64, 64), cells_per_block=(2, 2), visualize=True)
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))  # Rescale histogram for better display
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(A_rgb)
# plt.title('Image originale')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(hog_image_rescaled)
# plt.axis('off')

#  #%% KNN Classification using HOG feature
import seaborn as sns  # statistical data visualization library based on matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix  # metrics error
from sklearn.datasets import load_digits   # dataset for digit (0-9)
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split  # resampling method
digits = load_digits()  # load dataset # explore digits.keys()
plt.figure(2)
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.axis('off')

bins = 6
px_cel = 2
cl_blc = 2
N = len(digits.target)
fd = hog(digits.images[0], bins, pixels_per_cell=(px_cel, px_cel), cells_per_block=(cl_blc, cl_blc))
fd_size = len (fd)
fd = np.empty((N, fd_size))
for i in range(N):
    fd[i, :] = hog(digits.images[i], bins, pixels_per_cell=(px_cel, px_cel), cells_per_block=(cl_blc, cl_blc))

X = fd   #X = digits.data #for color descriptor
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
#knn = OneVsRestClassifier(KNeighborsClassifier(5))
# Fit the k-nearest neighbors classifier from the training dataset.
knn.fit(X_train, y_train)
# Predict the class labels for the provided data.
predictions = knn.predict(X_test)
print('KNN Accuracy: %.2f percents ' %(100*accuracy_score(y_test, predictions)))
cm = confusion_matrix(y_test, predictions)

plt.figure(3, figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5,square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}%'.format(100*accuracy_score(y_test, predictions))
plt.title(all_sample_title, size=15)

# %% Matching using SIFT Descriptor
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# #from skimage import data
# from skimage import transform
# from skimage.feature import match_descriptors, plot_matches, SIFT
# plt.close('all')

# image_name = 'fge.jpeg'
# A_bgr = cv2.imread(image_name)
# A_rgb = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB);
# img1 = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)

# img2 = transform.rotate(img1, 45)
# tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -100))
# img3 = transform.warp(img1, tform)

# desc = SIFT()

# desc.detect_and_extract(img1)
# keypoints1 = desc.keypoints; 
# descriptors1 = desc.descriptors

# desc.detect_and_extract(img2)
# keypoints2 = desc.keypoints; descriptors2 = desc.descriptors

# desc.detect_and_extract(img3)
# keypoints3 = desc.keypoints; descriptors3 = desc.descriptors

# matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)
# matches13 = match_descriptors(descriptors1, descriptors3, max_ratio=0.6, cross_check=True)

# # Features matching vizualisation
# fig = plt.figure(6)
# fig, ax = plt.subplots(2,2)

# #plt.gray()
# plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12);ax[0, 0].axis('off')
# ax[0, 0].set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

# plot_matches(ax[1, 0], img1, img3, keypoints1, keypoints3, matches13); ax[1, 0].axis('off')
# ax[1, 0].set_title(    "Original Image vs. Transformed Image\n" "(all keypoints and matches)")

# SUB1,_= np.shape(matches12); SUB1=SUB1//10 ; SUB2,_= np.shape(matches13); SUB2=SUB2//10
# plot_matches(ax[0, 1], img1, img2, keypoints1, keypoints2, matches12[::SUB1], only_matches=True); ax[0, 1].axis('off')
# ax[0, 1].set_title("Original Image vs. Flipped Image\n" "(10 percents subset of matches for visibility)")
# plot_matches(ax[1, 1], img1, img3, keypoints1, keypoints3, matches13[::SUB2], only_matches=True);ax[1, 1].axis('off')
# ax[1, 1].set_title("Original Image vs. Transformed Image\n" "(10 percents subset of matches for visibility)")

# # #plt.tight_layout()
