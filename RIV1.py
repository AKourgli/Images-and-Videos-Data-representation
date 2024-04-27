# -*- coding: utf-8 -*-

'''OpenCV is a popular open-source package that covers a wide range of image processing and computer vision capabilities
 and methods. It supports multiple programming languages including Python, C++, and Java
To install :  pip install opencv-python''' 
 
#from PIL import Image
import cv2
#import numpy as np
import matplotlib.pyplot as plt
 
# Load the image
A = cv2.imread('fge.jpeg')
#A = cv2.imread('USTHB.BMP')


# show image

plt.close('all'); 
# plt.figure(1); plt.subplot(231) ; plt.imshow(A); plt.title('BGR')

# # Convert BGR image to RGB
# A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
# plt.subplot(232); plt.title('RGB'); plt.imshow(A_rgb)

# # Convert BGR image to RGB
# A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
# #AA_gray = 0.299*A[:,:,2]+0.587*A[:,:,1]+0.114*A[:,:,0]
# plt.subplot(233); plt.title('Gray Levels'); plt.imshow(A_gray,cmap='gray')

# #save image
# cv2.imwrite('fge_gray.jpeg', A_gray)

# # show 3 RGB bands
# plt.subplot(234)
# plt.imshow(A_rgb[:,:,0],cmap='gray'); plt.title('Red')
# plt.subplot(235)
# plt.imshow(A_rgb[:,:,1],cmap='gray'); plt.title('Green')
# plt.subplot(236)
# plt.imshow(A_rgb[:,:,2],cmap='gray'); plt.title('Blue')

# # Hue, Saturation, Lightness : Teinte, saturation, luminosité (ou valeur)
# # Hue : degree on the color wheel from 0 to 360. 0 (or 360) is red, 120 is green, 240 is blue.
# # Saturation :  intensity of a color. It is a percentage value from 0% to 100% (100% is full color, no shades of gray, 50% is 50% gray, but you can still see the color; 0% is completely gray; you can no longer see the color.
# # Lightness/Value : how much light you want to give the color, where 0% means no light (dark), 50% means 50% light (neither dark nor light), and 100% means full light.
# A_HSV = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)

# plt.figure(2)
# plt.subplot(231); plt.imshow(A_rgb);plt.title('RGB');
# plt.subplot(232); plt.imshow(A_HSV); plt.title('HSV');
# plt.subplot(233); plt.imshow(A_gray,cmap='gray'); plt.title('Gray Levels');
# plt.subplot(234); plt.imshow(A_HSV[:,:,0],cmap='hsv'); plt.title('H');
# plt.subplot(235); plt.imshow(A_HSV[:,:,1],cmap='gray'); plt.title('S');
# plt.subplot(236); plt.imshow(A_HSV[:,:,2],cmap='gray'); plt.title('V');
# BB=(A_HSV[:,:,0])


# # Y : Luminance or the Luma. Y = 0.3 R + 0.6 V + 0.1 B
# # Cr : R - Y difference between the R color channel of RGB color space and the luminance component. 
# # Cb : B - Y difference between the B color channel of RGB color space and the luminance component.

# A_YCrCb = cv2.cvtColor(A, cv2.COLOR_BGR2YCrCb)
# plt.figure(3)
# plt.subplot(231); plt.imshow(A_rgb);plt.title('RGB');
# plt.subplot(232); plt.imshow(A_YCrCb); plt.title('HSV');
# plt.subplot(233); plt.imshow(A_gray,cmap='gray'); plt.title('Gray Levels');
# plt.subplot(234); plt.imshow(A_YCrCb[:,:,0],cmap='gray'); plt.title('Y');
# plt.subplot(235); plt.imshow(A_YCrCb[:,:,1],cmap='Reds'); plt.title('Cr');
# plt.subplot(236); plt.imshow(A_YCrCb[:,:,2],cmap='Blues'); plt.title('Cb');


import time
B = cv2.VideoCapture('usthb.mp4')
# fps = B.get(cv2.CAP_PROP_FPS)
# print('frames per second =',fps)

count = 0

while (True):
    ret, frame = B.read();       #read frames 
    time.sleep(1/15)            #pause
    #cv2.imwrite("USTHB_frame_{}.png".format(count), frame)
    cv2.imshow('USTHB',frame)   #show frame
    count +=1
    if(cv2.waitKey(1) & 0xFF == ord('q')): break        #quit if 'q' # 0xFF = 11111111 
 
#quit and close all opened windows
B.release()
cv2.destroyAllWindows()


B = cv2.VideoCapture('usthb.mp4')
w = int(B.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(B.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(B.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('usthb.avi', fourcc, fps, (w, h))


while B.isOpened():
    ret, frame = B.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):    break

# Release everything if job is finished
B.release()
out.release()
cv2.destroyAllWindows()







#Header access
# import exifread

# # Open image file for reading (binary mode)
# f = open('fge.jpeg', 'rb')

# # Return Exif tags
# tags = exifread.process_file(f)

# # Print the tags
# for tag in tags.keys():
#     print(tag, tags[tag])

# from PIL import Image
# from PIL.ExifTags import TAGS
# # open the image
# image = Image.open('fge.jpeg')
# # extracting the exif metadata
# exifdata = image.getexif()
 
# # looping through all the tags present in exifdata
# for tagid in exifdata:
#     # getting the tag name instead of tag id
#     tagname = TAGS.get(tagid, tagid)
#     # passing the tagid to get its respective value
#     value = exifdata.get(tagid).decode("utf-16")
#     # printing the final result
#     print(f"{tagname:25}: {value}")

