from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train5/weights/best.pt")
    results = model.val(data="config.yaml", name="val_results")
    print(results)