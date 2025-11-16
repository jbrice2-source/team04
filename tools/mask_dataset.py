import os
from glob import glob
import random
import cv2
import numpy as np

images = glob("MiRo Image Masking Dataset/Image/*")
images2 = glob("MiRo Image Masking Dataset/image_2/*")

random.shuffle(images)
random.shuffle(images2)
images += images2
# print(images[:3])

if not os.path.exists("mask_dataset"):
    os.mkdir("mask_dataset")
    os.mkdir("mask_dataset/train")
    os.mkdir("mask_dataset/test")
    os.mkdir("mask_dataset/train/images")
    os.mkdir("mask_dataset/train/labels")
    os.mkdir("mask_dataset/test/images")
    os.mkdir("mask_dataset/test/labels")

count = 0

for n in images[:7*len(images)//10]:
    img_name = os.path.basename(n).split(".")[0]
    print(img_name, count)
    count += 1
    if n.split("/")[1] == "image_2":
        image = cv2.imread(f"MiRo Image Masking Dataset/image_2/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"MiRo Image Masking Dataset/Mask_3/{img_name}.png",cv2.IMREAD_GRAYSCALE)
        img_name = "2"+img_name
    else:
        image = cv2.imread(f"MiRo Image Masking Dataset/Image/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"MiRo Image Masking Dataset/Mask_2/{img_name}.png",cv2.IMREAD_GRAYSCALE)



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
        perimeter = cv2.arcLength(max_cont, True)
        new_contours = cv2.approxPolyDP(max_cont, 0.002 * perimeter, True)
        cont_img = cv2.drawContours(image.copy(), [max_cont], -1, (0,255,0), 3)
        # cv2.imshow("contour mask",cont_img)
        cont_imgs = cv2.drawContours(image.copy(), [new_contours], -1, (0,255,0), 3)
        # cv2.imshow("simp contour mask",cont_imgs)
        
        # cv2.imshow("mask",mask)
        # print(max_cont.shape)
        norm_conts = new_contours.reshape(-1,2)/np.array(mask.shape[::-1])
        cont_list = " ".join(str(e) for e in norm_conts.flatten())
        # print(f"0 {cont_list}", norm_conts, mask.shape)
        # cv2.waitKey(0)
        cv2.imwrite(f"mask_dataset/train/images/{img_name}.jpg", image)
        
        with open(f"mask_dataset/train/labels/{img_name}.txt","w") as file:
            file.write(f"0 {cont_list}")
    else:
        cv2.imwrite(f"mask_dataset/train/images/{img_name}.jpg", image)
        
        with open(f"mask_dataset/train/labels/{img_name}.txt","w") as file:
            file.write("")
        
        
for n in images[7*len(images)//10:]:
    img_name = os.path.basename(n).split(".")[0]
    if n.split("/")[1] == "image_2":
        image = cv2.imread(f"MiRo Image Masking Dataset/image_2/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"MiRo Image Masking Dataset/Mask_3/{img_name}.png",cv2.IMREAD_GRAYSCALE)
        img_name = "2"+img_name
    else:
        image = cv2.imread(f"MiRo Image Masking Dataset/Image/{img_name}.jpg",cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(f"MiRo Image Masking Dataset/Mask_2/{img_name}.png",cv2.IMREAD_GRAYSCALE)


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
        perimeter = cv2.arcLength(max_cont, True)
        new_contours = cv2.approxPolyDP(max_cont, 0.002 * perimeter, True)
        cont_img = cv2.drawContours(image.copy(), [max_cont], -1, (0,255,0), 3)
        # cv2.imshow("contour mask",cont_img)
        cont_imgs = cv2.drawContours(image.copy(), [new_contours], -1, (0,255,0), 3)
        # cv2.imshow("simp contour mask",cont_imgs)
        
        # cv2.imshow("mask",mask)
        # print(max_cont.shape)
        norm_conts = new_contours.reshape(-1,2)/np.array(mask.shape[::-1])
        cont_list = " ".join(str(e) for e in norm_conts.flatten())
        # print(f"0 {cont_list}", norm_conts, mask.shape)
        # cv2.waitKey(0)
        cv2.imwrite(f"mask_dataset/test/images/{img_name}.jpg", image)
        
        with open(f"mask_dataset/test/labels/{img_name}.txt","w") as file:
            file.write(f"0 {cont_list}")
    else:
        cv2.imwrite(f"mask_dataset/test/images/{img_name}.jpg", image)
        
        with open(f"mask_dataset/test/labels/{img_name}.txt","w") as file:
            file.write("")
        

