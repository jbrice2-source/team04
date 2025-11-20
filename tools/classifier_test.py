from ultralytics import YOLO
import onnxruntime
import cv2
import numpy as np

model = YOLO("runs/classify/train26/weights/best.pt")  # load a custom model

# Predict with the model
results = model("MiRo Image Masking Dataset/Image/image398.jpg")  # predict on an image

# results = model("neg_dataset/eb00dd16efae4f3faefb3ff92463b37c.png")  # predict on an image
# onnx_model = onnxruntime.InferenceSession("runs/classify/train15/weights/best.onnx")

#["345_15","15_45","45_75","75_105", "105_135","135_165","165_195","195_225","225_255","255_285","285_315","315_345"]

# img = cv2.imread("neg_dataset/eb00dd16efae4f3faefb3ff92463b37c.png")


# img_w, img_h = img.shape[1], img.shape[0]

# img = cv2.resize(img, (640, 640))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img.transpose(2, 0, 1)
# img = img.reshape(1, 3, 640, 640)

# img = img/255.0
# img = img.astype(np.float32)
# # Run inference
# outputs = onnx_model.run(None, {"images": img})
# print(outputs)