from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/classify/train19/weights/best.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/classifier_dataset",
                      hsv_v=0.7, degrees=30, translate=0.2, scale=0.6,shear=5, perspective=0.0002, cutmix=0.1, device='cuda', epochs=100, imgsz=640)
