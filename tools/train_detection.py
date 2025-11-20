from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/detect/train29/weights/last.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/detection_yolo.yaml", device="cuda", cache=True, batch=24, epochs=600, imgsz=640,
                                hsv_v=0.7, degrees=40, translate=0.2, scale=0.6,shear=5, perspective=0.0002, cutmix=0.1)
