import cv2
import numpy as np
from glob import glob
import os

for i in glob("detection_dataset/test/images/*")[:]:
    image = cv2.imread(i,cv2.IMREAD_COLOR_RGB)
    # image = cv2.resize(image, (640,320))
    with open(f"detection_dataset/test/labels/{os.path.basename(i).split('.')[0]}.txt") as file:
        params = file.read().strip().split(" ")

    h,w = image.shape[:2]

    cx = 80
    cy = 240
    cx = np.random.randint(w//4,3*w//4)
    cy = np.random.randint(h//4,3*h//4)


    fx = 8000
    fy = 8000


    cam_matrix = np.array([ [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
        
    ])
    distort_coefficient = np.array([-.5,-.5,-.1,-.1])
    distort_coefficient = np.append(np.random.rand(2)*-1.8-.8,(np.random.rand(2)-0.5)*1.6)
    # print(cx,cy,distort_coefficient)


    img2 = cv2.undistort(image, cam_matrix,distort_coefficient)
    params = list(map(float, params))


    bbox = np.round(np.array([(params[1]-params[3]/2)*w,(params[2]-params[4]/2)*h,(params[1]+params[3]/2)*w,(params[2]+params[4]/2)*h])).astype(int)
    coords = np.array([[bbox[0],bbox[1]],[bbox[0],bbox[3]],[bbox[2],bbox[1]],[bbox[2],bbox[3]]])

    # normalised_coords = (coords - np.array([[cx,cy]]))/np.array([[fx,fy]]).astype(np.float32)
    points = cv2.undistortPoints(coords.astype(np.float32).reshape(-1,1,2), cam_matrix, distort_coefficient)

    normalised_points = points*np.array([[fx,fy]]) + np.array([[cx,cy]])
    normalised_points = np.round(normalised_points).astype(int).reshape(-1,2)


    # kernel = np.ones((5,5),np.float32)/20
    # img2 = cv2.filter2D(img2,-1,kernel)
    mean = 0
    stddev = np.repeat(30,3)
    noise = np.zeros(img2.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    # Add noise to image
    img2 = cv2.add(img2, noise)
    new_bbox = np.array([np.min(normalised_points,axis=0),np.max(normalised_points,axis=0)])
    centroid = np.clip(np.mean(new_bbox,axis=0)/np.array([w,h]),0,1)
    dimentions = np.clip(np.diff(new_bbox,axis=0)/np.array([w,h]),0,1).reshape(2)


    # cv2.rectangle(image,coords[0],coords[1],(255,0,0))
    # cv2.rectangle(img2,new_bbox[0],new_bbox[1],(255,0,0))
    

    # print(f"{params} {centroid[0]} {centroid[1]} {dimentions[0]} {dimentions[1]}")
    # print( f"{int(params[0])} {centroid[0]} {centroid[1]} {dimentions[0]} {dimentions[1]}")
    # cv2.imshow("image", img2)
    # cv2.waitKey(0)
    
    cv2.imwrite(f"detection_dataset/test/images/{os.path.basename(i).split('.')[0]}_v2.jpg",img2)
    with open(f"detection_dataset/test/labels/{os.path.basename(i).split('.')[0]}_v2.txt", "w") as file:
        file.write(f"{int(params[0])} {centroid[0]} {centroid[1]} {dimentions[0]} {dimentions[1]}")