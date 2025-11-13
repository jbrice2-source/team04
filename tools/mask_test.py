
from ultralytics import YOLO
import random
import cv2
import numpy as np

model_file = "runs/segment/train7/weights/best.pt"
model = YOLO(model_file)

image = cv2.imread("MiRo Image Masking Dataset/image_2/image385.jpg")
image = cv2.imread("pictures/90_1.png",cv2.IMREAD_COLOR_RGB)
# image = cv2.imread("processed_dataset/test/images/6b84b4f5bf4d4ca284661013fac604d2.png",cv2.IMREAD_COLOR_RGB)

# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

conf = 0.5

results = model.predict(image)
# colors = [random.choices(range(256), k=3) for _ in classes_ids]
# print(results)
print(len(results[0].masks.xy))
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        # cv2.polylines(image, points, True, (255, 0, 0), 1)
        # color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(image, points, (0,0,255))

cv2.imshow("Image", image)
cv2.waitKey(0)
