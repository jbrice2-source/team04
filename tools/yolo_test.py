from ultralytics import YOLO

# Load the best model we have so far:
model_file = "runs/detect/train/weights/best.pt"
model = YOLO(model_file)


import cv2

image = cv2.imread("dataset/0bb1fd3d6d3d46338038a9cf084d8a26.png")

# remember to convert it to RGB if using OpenCV
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# predict returns a list of Results object. 
# Since we are running on a single image, we'll take the only one result
results = model.predict(rgb)[0]
print(results.boxes.xyxy)
print(results.boxes.xywh)