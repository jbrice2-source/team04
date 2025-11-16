from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(data="/home/hancom/miro-docker/team04/mask_yolo.yaml", epochs=10, imgsz=640)
