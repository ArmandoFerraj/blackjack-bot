from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train5/weights/best.pt")
    results = model.predict(source="C:/Users/aferr/Desktop/bj_bot/data/test/images", save=True, save_txt=True, name="test_results")
    print(results)