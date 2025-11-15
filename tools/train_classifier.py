from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/classify/train13/weights/last.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/classifier_dataset", device='cuda', epochs=50, imgsz=640)
