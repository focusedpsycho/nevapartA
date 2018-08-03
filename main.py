import cv2
import numpy as np
import os
#enter input folder name
source_dir = 'images2'
for filename in os.listdir(source_dir):
    original_img = cv2.imread('images2/'+filename, cv2.IMREAD_COLOR) #enter input folder name
    width, height, depth = original_img.shape # get image dimensions
    gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  # convert rgb to gray scale
    median_filtered = cv2.medianBlur(gray_image, 5) # median filter to remove noise
    ret, binary_image = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU's thresholding
    kernel = np.ones((width // 12, height // 12), np.uint8) # kernel for opening
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel) #opening- erosion follwed by dilution
    erosion = cv2.erode(opening, np.ones((1, height), np.uint8)) #erode using a single line to shape it into a rect
    image, contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
    cnt = contours[0] #extract contours
    x, y, w, h = cv2.boundingRect(cnt) # get coordinates of bounding rectangle
    crop = original_img[y:y + h, x:x + w] # crop according to bounding rectangle
    cv2.imwrite('output2/'+filename, crop) # specify output folder

