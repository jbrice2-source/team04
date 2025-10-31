import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from glob import glob
 
MIN_MATCH_COUNT = 10
images = glob("cropped_pictures/*.png")[::2]
img2 = cv.imread('pictures/135_1.png', cv.IMREAD_GRAYSCALE) # trainImage
best = ([],None,None,None)
 
for i in images:

    img1 = cv.imread(i, cv.IMREAD_GRAYSCALE)          # queryImage

    
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good) > len(best[0]):
        best = (good,img1,kp1,kp2)
        print(i)
        
        
if len(best[0])>MIN_MATCH_COUNT:
    src_pts = np.float32([ best[2][m.queryIdx].pt for m in best[0] ]).reshape(-1,1,2)
    dst_pts = np.float32([ best[3][m.trainIdx].pt for m in best[0] ]).reshape(-1,1,2)
 
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
 
    h,w = best[1].shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    # img2 = cv.polylines(img2,[np.int32([[[100,100]],[[100,200]],[[200,100]],[[200,100]]])],True,255,3, cv.LINE_AA)
    centroid = np.sum(dst, axis=0)/dst.shape[0]
    img2 = cv.circle(img2, (round(centroid[0,0]),round(centroid[0,1])), 10, (128,0,0), -1)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
 
img3 = cv.drawMatches(best[1],best[2],img2,best[3],best[0],None,**draw_params)
 
plt.imshow(img3, 'gray'),plt.show()