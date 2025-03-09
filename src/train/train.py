from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # Start with pretrained model
    model.train(data="data.yaml", epochs=100, imgsz=640, batch=32)