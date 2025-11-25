import os
from glob import glob
import random
import cv2
import numpy as np

images_classes = glob("Labelled_dataset/Image/*")
# images2 = glob("MiRo Image Masking Dataset/image_2/*")

random.shuffle(images_classes)
# random.shuffle(images2)
# images += images2
# print(images[:3])

if not os.path.exists("detection_dataset"):
    os.mkdir("detection_dataset")
    os.mkdir("detection_dataset/train")
    os.mkdir("detection_dataset/test")
    os.mkdir("detection_dataset/train/images")
    os.mkdir("detection_dataset/train/labels")
    os.mkdir("detection_dataset/test/images")
    os.mkdir("detection_dataset/test/labels")
    
image_class_names = ["0", "45", "90", "135", "180", "225", "270", "315"]

count = 0
for i in images_classes:
    images = glob(f"{i}/*")
    class_name = i.split("/")[-1]
    for n in images[:8*len(images)//10]:
        img_name = os.path.basename(n).split(".")[0]
        print(img_name, count)
        count += 1
        image = cv2.imread(f"Labelled_dataset/Image/{class_name}/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"Labelled_dataset/Mask_2/{img_name}.png",cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((5,5))
        smooth = cv2.dilate(mask, kernel,iterations=2)
        smooth = cv2.erode(mask,kernel,iterations=2)
        # cv2.imshow("dilate",smooth)

        ret, thresh = cv2.threshold(smooth, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("image",image)
        # print(len(contours))
        max_cont = np.array([])
        for i in contours:
            if len(max_cont) < len(i):
                max_cont = i
        if len(max_cont) > 10:
            x,y, width, height = cv2.boundingRect(max_cont)
            img_height, img_width = image.shape[:2]
            centre = [x+width/2,y+height/2]
            cv2.imwrite(f"detection_dataset/train/images/{img_name}.jpg", image)
            
            with open(f"detection_dataset/train/labels/{img_name}.txt","w") as file:
                file.write(f"{image_class_names.index(class_name)} {centre[0]/img_width} {centre[1]/img_height} {width/img_width} {height/img_height}")
        else:
            cv2.imwrite(f"detection_dataset/train/images/{img_name}.jpg", image)
            
            
    for n in images[8*len(images)//10:]:
        img_name = os.path.basename(n).split(".")[0]
        # if n.split("/")[1] == "image_2":
        #     image = cv2.imread(f"MiRo Image Masking Dataset/image_2/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        #     mask = cv2.imread(f"MiRo Image Masking Dataset/Mask_3/{img_name}.png",cv2.IMREAD_GRAYSCALE)
        #     img_name = "2"+img_name
        # else:
        image = cv2.imread(f"Labelled_dataset/Image/{class_name}/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"Labelled_dataset/Mask_2/{img_name}.png",cv2.IMREAD_GRAYSCALE)


        # kernel = np.ones((5,5))
        # smooth = cv2.dilate(mask, kernel,iterations=2)
        # smooth = cv2.erode(mask,kernel,iterations=2)
        # cv2.imshow("dilate",smooth)

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("image",image)
        # print(len(contours))
        max_cont = np.array([])
        for i in contours:
            if len(max_cont) < len(i):
                max_cont = i
        if len(max_cont) > 10:
            x,y, width, height = cv2.boundingRect(max_cont)
            img_height, img_width = image.shape[:2]
            centre = [x+width/2,y+height/2]
            cv2.imwrite(f"detection_dataset/test/images/{img_name}.jpg", image)
            
            with open(f"detection_dataset/test/labels/{img_name}.txt","w") as file:
                file.write(f"{image_class_names.index(class_name)} {centre[0]/img_width} {centre[1]/img_height} {width/img_width} {height/img_height}")
        else:
            cv2.imwrite(f"detection_dataset/test/images/{img_name}.jpg", image)
            

