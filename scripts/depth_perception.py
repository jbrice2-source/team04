import cv2
import numpy as np
# Read left and right images
left_image = cv2.imread('pictures/270_0.png')
# left_image = cv2.imread('wall3.jpg')
right_image = cv2.imread('pictures/270_1.png')
# right_image = cv2.imread('wall2.jpg')

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
print(150/points2[:,0])
print(1/points2[:,1])
dists = 150/points2[:,0]

dist_img = left_image.copy()
for i,n in enumerate(points):
    if dists[i] < 0.55:
        cv2.circle(dist_img,np.astype(n[0],int),5,(64,0,0))
    elif dists[i] < 0.6:
        cv2.circle(dist_img,np.astype(n[0],int),5,(128,0,0))
    elif dists[i] < 0.65:
        cv2.circle(dist_img,np.astype(n[0],int),5,(192,0,0))
    elif dists[i] < 0.7:
        cv2.circle(dist_img,np.astype(n[0],int),5,(256,0,0))

cv2.imshow('Left Image', orb_img)
cv2.imshow('Right Image', orb_img2)
cv2.imshow('matches Image', img3)
cv2.imshow('matches Image', dist_img)

cv2.waitKey(0)


cv2.destroyAllWindows()