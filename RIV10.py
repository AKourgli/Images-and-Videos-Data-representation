# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:35:44 2024

@author: AKourgli
"""
#%% 
#https://www.linkedin.com/pulse/motion-detection-using-python-dominic-oladapo-tonade/
 
# import cv2  as cv
# #import numpy as np

# #video = cv.VideoCapture("Bangkok2.mov") # video = cv2.VideoCapture("shibuya.mp4")   video = cv2.VideoCapture("usthb.mp4")
# #video = cv.VideoCapture("Paris.mov")   #video = cv.VideoCapture("kiev.mp4")
# video = cv.VideoCapture("usthb.mp4")
# ret, cur_frame = video.read()  
# gray_image = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)  
# First_frame = cv.GaussianBlur(gray_image, (25, 25), 0)  

# while True:  
#     ret, cur_frame = video.read()  
#     gray_image = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)  
#     gray_frame = cv.GaussianBlur(gray_image, (25, 25), 0)  
 
#     diff_frame = cv.absdiff(First_frame, gray_frame)       #diff_frame = cv.medianBlur(diff_frame, 3)
#     thresh_frame = cv.threshold(diff_frame, 50, 255, cv.THRESH_BINARY)[1]  
#     thresh_frame = cv.dilate(thresh_frame, None, iterations = 1) 
#     cont,_ = cv.findContours(thresh_frame.copy(),cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #second is contour retrieval mode, third is contour approximation method.
#     for cur in cont:  
#         if cv.contourArea(cur) < 5000:  
#             continue  
#         (cur_x, cur_y,cur_w, cur_h) = cv.boundingRect(cur)  
#         cv.rectangle(cur_frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 3)  
    
#     cv.namedWindow("Processed Frame Smoothed with a Gaussian Filter", cv.WINDOW_NORMAL) ; cv.imshow("Processed Frame Smoothed with a Gaussian Filter", gray_frame)
#     cv.namedWindow("Difference between the  inital static frame and the current frame", cv.WINDOW_NORMAL) ;cv.imshow("Difference between the  inital static frame and the current frame", diff_frame)  
#     cv.namedWindow("Threshold Frame created", cv.WINDOW_NORMAL) ;cv.imshow("Threshold Frame created", thresh_frame)  
#     cv.namedWindow("Cureent frame with bounding boxes", cv.WINDOW_NORMAL) ;cv.imshow("Cureent frame with bounding boxes", cur_frame) 
    
#     #First_frame = gray_frame
#     wait_key = cv.waitKey(1)  
#     if wait_key == ord('q'):  
#         break 
    
# video.release() 
# cv.destroyAllWindows()
# https://medium.com/@itberrios6/introduction-to-motion-detection-part-1-e031b0bb9bb2
#%%

# import cv2 as cv
# import numpy as np

# # cap = cv.VideoCapture("shibuya.mp4") #cap = cv.VideoCapture("usthb.mp4")  
# # cap = cv.VideoCapture("Bangkok2.mov") 
# cap = cv.VideoCapture("kiev.mp4")
# ret, first_frame = cap.read()
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# mask = np.zeros_like(first_frame)
# mask[..., 1] = 255   # Sets image saturation to maximum

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv.imshow("Processed Video", frame)
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Calculates dense optical flow by Farneback method
#     flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 50, 3, 5, 1.2, 0)
#     magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) # Computes the magnitude and angle of the 2D vectors
#     mask[..., 0] = angle * 180 / np.pi / 2  # Sets image hue according to the optical flow direction
#     mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) # Sets image value according to the optical flow magnitude (normalized)
#     rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR) # Converts HSV to RGB (BGR) color representation
#     cv.imshow("Dense optical flow", rgb) # Opens a new window and displays the output frame
#     prev_gray = gray  # Updates previous frame
#     #cv.imshow("Processed Video with otical flow", frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()

#%%

import cv2 as cv
import numpy as np

#cap = cv.VideoCapture("shibuya.mp4")  # cap = cv.VideoCapture("usthb.mp4") # cap = cv.VideoCapture("Kiev.mp4") # cap = cv.VideoCapture("Paris.mov") #cap = cv.VideoCapture("Autoroute.mp4") 
cap = cv.VideoCapture("Bangkok2.mov") 
color1 = (0, 255, 0) ; color2 = (0, 0, 255) # Variable for color to draw optical flow track
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive

feature_params = dict(maxCorners = 100, qualityLevel = 0.5, minDistance = 2, blockSize = 7) # Parameters for Shi-Tomasi corner detection
lk_params = dict(winSize = (5,5), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))  # Parameters for Lucas-Kanade optical flow

mask = np.zeros_like(first_frame)  # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params) # Finds the strongest corners in the first frame by Shi-Tomasi method for which the optical flow will be tracked
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    good_old = prev[status == 1].astype(int) # Selects good feature points for previous position
    good_new = next[status == 1].astype(int) # Selects good feature points for next position
    
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Returns a contiguous flattened array as (x, y) coordinates for new point
        c, d = old.ravel()  # Returns a contiguous flattened array as (x, y) coordinates for old point
        mask = cv.line(mask, (a, b), (c, d), color1, 3) # Draws line between new and old position with green color and 2 thickness
        frame = cv.circle(frame, (a, b), 5, color2, 1) # Draws filled circle (thickness of -1) at new position with red color and radius of 5
    output = cv.add(frame, mask)  # Overlays the optical flow tracks on the original frame
    prev_gray = gray.copy() # Updates previous frame
    prev = good_new.reshape(-1, 1, 2)  # Updates previous good feature points
    
    cv.namedWindow("sparse optical flow",cv.WINDOW_NORMAL) # Opens a new window and displays the output frame
    cv.imshow("sparse optical flow", output)
    if cv.waitKey(1) & 0xFF == ord('q'): # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        break

cap.release()  # The 2 following lines free up resources and closes all windows
cv.destroyAllWindows()
















'''
https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
Computes a dense optical flow using the Gunnar Farneback's algorithm.

Parameters
prev	    first 8-bit single-channel input image.
next	    second input image of the same size and the same type as prev.
flow	    computed flow image that has the same size as prev and type CV_32FC2.
pyr_scale	parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
levels	    number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
winsize	    averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
iterations	number of iterations the algorithm does at each pyramid level.
poly_n	    size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_sigma	standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
flags	    operation flags that can be a combination of the following:
OPTFLOW_USE_INITIAL_FLOW 
            uses the input flow as an initial flow approximation.
OPTFLOW_FARNEBACK_GAUSSIAN 
            uses the Gaussian 
 filter    instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.
 '''
 
'''
Parameters
image	       Input 8-bit or floating-point 32-bit, single-channel image.
corners	       Output vector of detected corners.
maxCorners	   Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
qualityLevel   Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
minDistance	   Minimum possible Euclidean distance between the returned corners.
mask	       Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
blockSize	   Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
useHarrisDetector
  	           Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal.
k	           Free parameter of the Harris detector.
'''


'''
Harris Corner Detector
https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5048786-detectez-les-coins-et-les-bords-dans-une-image
'''

'''
Parameters:
prevImg     first 8-bit input image
nextImg     second input image
prevPts     vector of 2D points for which the flow needs to be found.
winSize     size of the search window at each pyramid level.
maxLevel    0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on.
criteria    parameter, specifying the termination criteria of the iterative search algorithm.

Return:
nextPts    output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
status     output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
err        output vector of errors; each element of the vector is set to an error for the corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn’t found then the error is not defined (use the status parameter to find such cases).
'''