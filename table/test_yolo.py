from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.predict(
   source='detected_image.png',
   conf=0.25
)