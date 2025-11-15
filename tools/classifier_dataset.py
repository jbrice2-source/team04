
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
def func1(dataset,section):
    for d in dataset:
        
        img1 = cv2.imread(f"dataset/{d[0]}.png")      
        # sift.setUpright(True)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        # blur = cv2.bilateralFilter(img1, 9, 75,75)
        # cv2.imshow("bilatteral filter", blur)
        angle_classes = ["345_15","15_45","45_75","75_105","135_165","165_195","195_225","225_255","255_285","315_345"]

        if not os.path.isdir("classifier_dataset"):
            os.mkdir("classifier_dataset")
            os.mkdir("classifier_dataset/train")
            os.mkdir("classifier_dataset/test")
            for n in angle_classes+["background"]:
                os.mkdir(f"classifier_dataset/train/{n}")
                os.mkdir(f"classifier_dataset/test/{n}")

        num_data = list(map(float, d[1:]))
        # print(num_data)
        angle = np.arctan2(num_data[5]-num_data[2],num_data[4]-num_data[1])-num_data[3]    
        # print(angle, np.degrees(angle)%360)

        angle_class = len(angle_classes)*(round(np.degrees(angle)+15)%360)//360
        # print(angle_class,angle_classes[angle_class])
        # cv2.imshow("lines", img1)
        # cv2.imshow("contoured", edges)
        mean = 0
        stddev = 30
        noise = np.zeros(img1.shape[:2], np.uint8)
        cv2.randn(noise, mean, stddev)
        noise = cv2.merge([noise,noise,noise])
        # Add noise to image
        # print(noise.shape)
        noisy_img = cv2.add(img1, noise)
        
        # Get the image size (number of pixels in the image).
        img_size = noisy_img.size
        noise_percentage = 0.05  # Setting to 10%
        noise_size = int(noise_percentage*img_size)
        random_indices = np.random.choice(img_size, noise_size)
        img_noised = noisy_img.copy()
        noise = np.random.choice([noisy_img.min(), noisy_img.max()], noise_size)
        img_noised.flat[random_indices] = noise
        
        cv2.imwrite(f"classifier_dataset/{section}/{angle_classes[angle_class]}/{d[0]}.png", img_noised)
        # write_params = f"{angle_class}"
        # with open(f"classifier_dataset/{section}/labels/{d[0]}.txt", "w") as file:
        #     file.write(write_params)
        # cv2.waitKey(0)

func1(dataset[:len(dataset)*8//10],"train")
func1(dataset[len(dataset)*8//10:],"test")


background_images = glob("neg_dataset/*.png")
random.shuffle(background_images)
for i in background_images[:len(background_images)*8//10]:
    img1 = cv2.imread(i, cv2.IMREAD_COLOR_RGB)
    filename = os.path.basename(i).split(".")[0]
    mean = 0
    stddev = 30
    noise = np.zeros(img1.shape[:2], np.uint8)
    cv2.randn(noise, mean, stddev)
    noise = cv2.merge([noise,noise,noise])
    # Add noise to image
    noisy_img = cv2.add(img1, noise)
    # Get the image size (number of pixels in the image).
    img_size = noisy_img.size
    noise_percentage = 0.05  # Setting to 10%
    noise_size = int(noise_percentage*img_size)
    random_indices = np.random.choice(img_size, noise_size)
    img_noised = noisy_img.copy()
    noise = np.random.choice([noisy_img.min(), noisy_img.max()], noise_size)
    img_noised.flat[random_indices] = noise
    cv2.imwrite(f"classifier_dataset/train/background/{filename}.png", img_noised)
for i in background_images[len(background_images)*8//10:]:
    img1 = cv2.imread(i, cv2.IMREAD_COLOR_RGB)
    filename = os.path.basename(i).split(".")[0]
    mean = 0
    stddev = 30
    noise = np.zeros(img1.shape[:2], np.uint8)
    cv2.randn(noise, mean, stddev)
    noise = cv2.merge([noise,noise,noise])
    # Add noise to image
    noisy_img = cv2.add(img1, noise)
    # Get the image size (number of pixels in the image).
    img_size = noisy_img.size
    noise_percentage = 0.05  # Setting to 10%
    noise_size = int(noise_percentage*img_size)
    random_indices = np.random.choice(img_size, noise_size)
    img_noised = noisy_img.copy()
    noise = np.random.choice([noisy_img.min(), noisy_img.max()], noise_size)
    img_noised.flat[random_indices] = noise
    cv2.imwrite(f"classifier_dataset/test/background/{filename}.png", img_noised)