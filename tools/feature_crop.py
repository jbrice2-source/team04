
import cv2
import numpy as np
from glob import glob
from statistics import NormalDist
import os
import random
with open("dataset/labels.csv") as f:
    dataset = f.readlines()
    
dataset = list(e.strip().split(",") for e in dataset)

# images = glob("dataset/*.png")
random.shuffle(dataset)
sift = cv2.SIFT_create()
def func1(dataset,section):
    for d in dataset:
        
        img1 = cv2.imread(f"dataset/{d[0]}.png")      
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
        # print(x1,x2,y1,y2)
        # print(sizes)
        keypoint_image = cv2.drawKeypoints(img1,np.array(kp1)[sizes>=4.0],None,(255,0,0),4)
        padx = (x2-x1)//7
        pady = (y2-y1)//7
        bounding_image = img1.copy()
        cv2.rectangle(bounding_image,(x1-padx,y1-pady),(x2+padx,y2+pady), (0,0,255), 1)
        # cv2.imshow("processed edges", keypoint_image)
        # cv2.imshow("bounding image", bounding_image)
        bounding_box = np.array([max(y1-pady,0),min(y2+pady,img1.shape[0]),max(x1-padx,0),min(x2+padx,img1.shape[1])])
        # print(bounding_box)
        crop_image = img1[bounding_box[0]:bounding_box[1],bounding_box[2]:bounding_box[3]]
        # cv2.imshow("cropped image", crop_image)
        edges = cv2.Canny(img1, 100,200)
        kernel = np.ones((5,5))
        edges = cv2.dilate(edges,kernel, iterations=2)
        edges = cv2.erode(edges,kernel, iterations=2)
        # cv2.imshow("edges", edges)
        # gray_edges = cv2.cvtColor(edges,cv2.COLOR_RGB2GRAY)
        # linesP = cv2.HoughLinesP(edges, 2, np.pi/180, 80, None, 5, 30)
        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (0,0,0), 3, cv2.LINE_AA)
        #         cv2.line(img1, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        if not os.path.isdir("processed_dataset"):
            os.mkdir("processed_dataset")
            os.mkdir("processed_dataset/train")
            os.mkdir("processed_dataset/test")
            os.mkdir("processed_dataset/train/images")
            os.mkdir("processed_dataset/train/labels")
            os.mkdir("processed_dataset/test/images")
            os.mkdir("processed_dataset/test/labels")
            
        angle_classes = ["345-15","15-45","45-75","75-105","135-165","165-195","195-225","225-255","255-285","315-345"]

        num_data = list(map(float, d[1:]))
        print(num_data)
        angle = np.arctan2(num_data[5]-num_data[2],num_data[4]-num_data[1])-num_data[3]    
        print(angle, np.degrees(angle)%360)

        angle_class = len(angle_classes)*(round(np.degrees(angle)+15)%360)//360
        print(angle_class,angle_classes[angle_class])
        # cv2.imshow("lines", img1)
        # cv2.imshow("contoured", edges)
        mean = 0
        stddev = 30
        noise = np.zeros(img1.shape[:2], np.uint8)
        cv2.randn(noise, mean, stddev)
        noise = cv2.merge([noise,noise,noise])
        # Add noise to image
        print(noise.shape)
        noisy_img = cv2.add(img1, noise)

        cv2.imwrite(f"processed_dataset/{section}/images/{d[0]}.png", noisy_img)
        
        
        centre = np.array([bounding_box[2]+bounding_box[3],bounding_box[0]+bounding_box[1]])/(2*np.array(list(img1.shape[-2::-1]))) # np.mean(bounding_box.reshape(-1,2),axis=1)/np.array(list(img1.shape[-2::-1]))
        width = (bounding_box[3]-bounding_box[2])/img1.shape[1]
        height = (bounding_box[1]-bounding_box[0])/img1.shape[0]
        print(centre, width, height, img1.shape,np.array(list(img1.shape[-2::-1])))
        write_params = f"{angle_class} {centre[0]} {centre[1]} {width} {height}"
        with open(f"processed_dataset/{section}/labels/{d[0]}.txt", "w") as file:
            file.write(write_params)
        # cv2.waitKey(0)

func1(dataset[:len(dataset)*7//9],"train")
func1(dataset[len(dataset)*7//9:],"test")


background_images = glob("neg_dataset/*.png")
random.shuffle(background_images)
for i in background_images[:len(background_images)*7//9]:
    img1 = cv2.imread(i)
    filename = os.path.basename(i).split(".")[0]
    mean = 0
    stddev = 30
    noise = np.zeros(img1.shape[:2], np.uint8)
    cv2.randn(noise, mean, stddev)
    noise = cv2.merge([noise,noise,noise])
    # Add noise to image
    print(noise.shape)
    noisy_img = cv2.add(img1, noise)
    cv2.imwrite(f"processed_dataset/train/images/{filename}.png", noisy_img)
for i in background_images[len(background_images)*7//9:]:
    img1 = cv2.imread(i)
    filename = os.path.basename(i).split(".")[0]
    mean = 0
    stddev = 30
    noise = np.zeros(img1.shape[:2], np.uint8)
    cv2.randn(noise, mean, stddev)
    noise = cv2.merge([noise,noise,noise])
    # Add noise to image
    print(noise.shape)
    noisy_img = cv2.add(img1, noise)
    cv2.imwrite(f"processed_dataset/train/images/{filename}.png", noisy_img)