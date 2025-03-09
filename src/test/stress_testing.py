from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained model (replace trainX with your latest run)
    model = YOLO("best.pt")
    # Predict on test set and save annotated images
    model.predict(source="test2.jpg", save=True, save_txt=True, name="stress_test_results")