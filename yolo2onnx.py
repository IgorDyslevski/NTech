from ultralytics import YOLO


model = YOLO('models/yolov8n.pt')
model.export(format='onnx') #, dynamic=True)
