from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("runs/detect/train11/weights/last.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/settings.yaml", epochs=50, imgsz=640)
