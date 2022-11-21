import numpy as np
import cv2 as cv

# uses opencv's sift matching to automatically find matching points between two frames
# can be very helpful for locating 4+ points for calculating homography

# source - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

def readRGB(path, filename, width, height):
    f = open(path+filename, "r")
    arr = np.fromfile(f, np.uint8).reshape(height, width, 3)
    arr[:, :, [2, 0]] = arr[:, :, [0, 2]]
    return arr

img1 = readRGB("SAL_490_270_437/", "SAL_490_270_437.001.rgb", 490, 270)
img2 = readRGB("SAL_490_270_437/", "SAL_490_270_437.100.rgb", 490, 270)
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('frame', img3)
cv.waitKey(0);