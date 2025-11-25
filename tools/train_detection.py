from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/detect/train41/weights/last.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/detection_yolo.yaml", device="cuda", cache=True, batch=16, epochs=100, imgsz=640,
                                hsv_v=0.7, degrees=50, translate=0.4, scale=0.8, fliplr=0.0,shear=10, perspective=0.0004, cutmix=0.5)
