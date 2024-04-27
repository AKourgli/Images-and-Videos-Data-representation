# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:29:18 2024
https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_mean_shift_tracking_segmentation.php
"""
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt

roi = cv.imread('Image1.png'); roi_rgb=cv.cvtColor(roi, cv.COLOR_BGR2RGB);
target = cv.imread('Image2.png'); target_rgb=cv.cvtColor(target, cv.COLOR_BGR2RGB);


roi_hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
x=y=200; h=w=100; roi_hsv = roi_hsv[y:y+h,x:x+w,:]
target_hsv  = cv.cvtColor(target,cv.COLOR_BGR2HSV)

# roihist = cv.calcHist([roi_hsv],[0], None, [180], [0, 180] )
# cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
# dst = cv.calcBackProject([target_hsv],[0],roihist,[0,180],1)

# # calculating object histogram
roihist = cv.calcHist([roi_hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)

# Back Projection is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model.
# calculate the histogram model of a feature and then use it to find this feature in an image.
dst = cv.calcBackProject([target_hsv],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
# disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# cv.filter2D(dst,-1,disc,dst)

rec = cv.rectangle(roi_rgb, (x,y), (x+w,y+h), 255,2)

plt.figure(1)
plt.subplot(221); plt.imshow(roi_rgb); plt.axis('off')
plt.subplot(222); plt.imshow(target_rgb); plt.axis('off')
plt.subplot(223); plt.stem(roihist); 
plt.subplot(224); plt.imshow(dst,cmap='gray'); plt.axis('off')


"Tracking using meanshift"

cap = cv.VideoCapture('traffic.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        time.sleep(1/4)   
        cv.imshow('Tracking using meanshift',img2)

        if cv.waitKey(1) == ord('q'):    break

    else:
        break

"Tracking using camshift"

cap = cv.VideoCapture('traffic.mp4')
# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        time.sleep(1/4)  
        cv.imshow('Tracking using camshift',img2)
        if cv.waitKey(1) == ord('q'):    break

    else:
        break

''' https://docs.opencv.org/3.4.15/da/d7f/tutorial_back_projection.html
In terms of statistics, the values stored in BackProjection represent the probability
 that a pixel in Test Image belongs to the same area, based on the model histogram that we use.
 For instance in our Test image, the brighter areas are more probable to be our tracked area 
 (as they actually are), whereas the darker areas have less probability.'''

roihist = cv.calcHist([roi_hsv],[0], None, [180], [0, 180] )
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([target_hsv],[0],roihist,[0,180],1)
