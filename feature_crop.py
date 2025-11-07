
import cv2
import numpy as np
from glob import glob
from statistics import NormalDist
import argparse
images = glob("dataset/*.png")


sift = cv2.SIFT_create()

for n in images[190:200]:
    img1 = cv2.imread(n)      
    # sift.setUpright(True)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    # blur = cv2.bilateralFilter(img1, 9, 75,75)
    # cv2.imshow("bilatteral filter", blur)
    
    kp1, des1 = sift.detectAndCompute(img1,None)


    sizes = np.zeros_like(kp1)
    for i,n in enumerate(kp1):
        sizes[i] = n.size
        
    
    pnts = cv2.KeyPoint_convert(kp1)[sizes>=4.0]
    new_sizes = sizes[sizes>=4.0]
    mean = np.median(pnts,axis=0)
    dev = np.std(pnts,axis=0)
    
    print(pnts.shape,mean,dev)
    x1 = round(NormalDist(mean[0],dev[0]).inv_cdf(0.05))
    x2 = round(NormalDist(mean[0],dev[0]).inv_cdf(0.95))
    y1 = round(NormalDist(mean[1],dev[1]).inv_cdf(0.05))
    y2 = round(NormalDist(mean[1],dev[1]).inv_cdf(0.95))
    print(x1,x2,y1,y2)
    # print(sizes)
    keypoint_image = cv2.drawKeypoints(img1,np.array(kp1)[sizes>=4.0],None,(255,0,0),4)
    padx = (x2-x1)//5
    pady = (y2-y1)//5
    bounding_image = img1.copy()
    cv2.rectangle(bounding_image,(x1-padx,y1-pady),(x2+padx,y2+pady), (0,0,255), 1)
    cv2.imshow("processed edges", keypoint_image)
    cv2.imshow("bounding image", bounding_image)
    crop_image = img1[max(y1-pady,0):min(y2+pady,img1.shape[0]),max(x1-padx,0):min(x2+padx,img1.shape[1])]
    cv2.imshow("cropped image", crop_image)
    edges = cv2.Canny(img1, 100,200)
    kernel = np.ones((5,5))
    edges = cv2.dilate(edges,kernel, iterations=2)
    edges = cv2.erode(edges,kernel, iterations=2)
    cv2.imshow("edges", edges)
    # gray_edges = cv2.cvtColor(edges,cv2.COLOR_RGB2GRAY)
    linesP = cv2.HoughLinesP(edges, 2, np.pi/180, 80, None, 5, 30)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (0,0,0), 3, cv2.LINE_AA)
            cv2.line(img1, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    

    cv2.imshow("lines", img1)
    cv2.imshow("contoured", edges)

    cv2.waitKey(0)
