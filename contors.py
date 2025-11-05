import numpy as np
import cv2
from glob import glob

images = glob("pictures/*.png")

for n in images[:]:
    original_img = cv2.imread(n)
    img = original_img.copy()

    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # equalised_image = cv2.equalizeHist(imgray)
    # equalised_image = cv2.cvtColor(equalised_image, cv2.COLOR_GRAY2BGR)
    
    # img = cv2.add(img,equalised_image)
    # background = cv2.dilate(img,kernel,iterations=1)
    # img = cv2.subtract(background, img)
    # cv2.imshow("subtract", img)

    
    blur = cv2.bilateralFilter(img, 11, 75,75)
    cv2.imshow("bilatteral filter", blur)

    median = np.median(np.mean(original_img,axis=2))
    lower_threshold = int(max(0, 0.66 * median))
    upper_threshold = int(min(255, 1.33 * median))
    # blur = cv2.copyMakeBorder(blur,2,2,2,2,cv2.BORDER_CONSTANT,value=[255, 255, 255])

    edges = cv2.Canny(blur,lower_threshold,upper_threshold)


    edges = cv2.dilate(edges,kernel,iterations=2)
    edges = cv2.erode(edges,kernel,iterations=2)
    edges = cv2.bitwise_not(edges)

    cv2.imshow("processed edges", edges)

    ret, thresh = cv2.threshold(edges, 200, 255, 0,cv2.THRESH_BINARY)
    print(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    child_num = np.zeros(hierarchy.shape[1])
    for i in range(len(child_num)-1,-1,-1):
        if hierarchy[0,i,3] != -1:
            child_num[hierarchy[0,i,3]] += 1 + child_num[i]
            
    # print(child_num)
    # print(hierarchy)
    cont_list = []
    hiere_list = np.array([])
    for i,n in enumerate(contours):
        if len(n) < 500:
            cont_list.append(n)
            hiere_list = np.append(hiere_list,hierarchy[0,i]).reshape(-1,4)

    print(list(map(len,contours)))
    print(list(map(len,cont_list)))

    img2 = blur.copy()
    cv2.drawContours(img2, contours, -1, (0,255,0), 1)
    cv2.imshow('Contours raw', img2)
    # cv2.waitKey(0)

    # for i in cont_list:
    img3 = original_img.copy()
    cv2.drawContours(img3, cont_list, -1, (0,0,255), 1)
    cv2.imshow('Contours', img3)
    # cv2.waitKey(0)

    cont_max = np.array(list(map(lambda x:np.max(x,axis=0),cont_list))).reshape(-1,2)
    cont_min = np.array(list(map(lambda x:np.min(x,axis=0),cont_list))).reshape(-1,2)
    # print(cont_min)
    print(cont_min[:,0].min(),cont_max[:,0].max(),cont_min[:,1].max(),cont_max[:,1].max())
    cv2.imshow('Contours', img3)
    # cv2.waitKey(0)

    cv2.imshow("Contours crop",img3[max(cont_min[:,1].min()-30,0):min(cont_max[:,1].max()+30,img3.shape[1]),
                                    max(cont_min[:,0].min()-30,0):min(cont_max[:,0].max()+30,img3.shape[1])])
    cv2.waitKey(0)

cv2.destroyAllWindows()
