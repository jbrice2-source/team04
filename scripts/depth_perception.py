#!/usr/bin/env python3


import cv2
import numpy as np


with open("dataset/labels.csv") as f:
    dataset = f.readlines()
    
dataset = list(e.strip().split(",") for e in dataset)


# Read left and right images
left_image = cv2.imread(f"dataset/{dataset[16][0]}.png")
# left_image = cv2.imread('wall3.jpg')
right_image = cv2.imread(f"dataset/{dataset[17][0]}.png")
# right_image = cv2.imread('wall2.jpg')

print(np.sqrt(np.pow(float(dataset[6][2])-float(dataset[6][5]),2)+np.pow(float(dataset[6][3])-float(dataset[6][6]),2)))


# Display left and right images
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(left_image,None)
kp2, des2 = orb.detectAndCompute(right_image,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

print(des1.shape, len(kp1))
print(des1[0], des1[1])

orb_img = cv2.drawKeypoints(left_image,kp1,None, (255,0,0))
orb_img2 = cv2.drawKeypoints(right_image,kp2,None, (255,0,0))



# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

points = np.empty((len(good),2,2))


for i,n in enumerate(good):
    img1_idx = n[0].queryIdx
    img2_idx = n[0].trainIdx
    points[i,0] = kp1[img1_idx].pt
    points[i,1] = kp2[img2_idx].pt



# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(left_image,kp1,right_image,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# print(points)
points2 = np.zeros((points.shape[0],2))
for i,n in enumerate(points):
    points2[i] = n[0]-n[1]

# print(points2)
print((2*0.086*1000)/points2[:,0])
dists = (2*0.086*1000)/points2[:,0]

dist_img = left_image.copy()
for i,n in enumerate(points):
    if dists[i] < 0.6:
        cv2.circle(dist_img,np.astype(n[0],int),5,(64,255,0))
    elif dists[i] < 0.7:
        cv2.circle(dist_img,np.astype(n[0],int),5,(128,192,0))
    elif dists[i] < 0.8:
        cv2.circle(dist_img,np.astype(n[0],int),5,(192,128,0))
    else:
        cv2.circle(dist_img,np.astype(n[0],int),5,(255,64,0))

cv2.imshow('Left Image', orb_img)
cv2.imshow('Right Image', orb_img2)
cv2.imshow('matches Image', img3)
cv2.imshow('dist Image', dist_img)

cv2.waitKey(0)


cv2.destroyAllWindows()