from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")
    model.predict(source="test4.jpg", save=True, save_txt=True, name="stress_test_results")