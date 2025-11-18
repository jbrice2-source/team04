from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/classify/train24/weights/last.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/classifier_dataset",
                      hsv_v=0.7, degrees=40, translate=0.2, scale=0.6,shear=5, perspective=0.0003, fliplr=0.0, cutmix=0.1, batch=-1, cache=True, device='cuda', epochs=50, imgsz=640)
