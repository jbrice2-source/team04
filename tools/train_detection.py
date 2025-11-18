from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11s.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/detection_yolo.yaml", device="cuda", epochs=1, imgsz=480,
                                hsv_v=0.7, degrees=30, translate=0.2, scale=0.6,shear=5, perspective=0.0002, cutmix=0.1)
