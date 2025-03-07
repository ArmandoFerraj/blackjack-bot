from ultralytics import YOLO

if __name__ == '__main__':
    # Load the model
    model = YOLO("../src/models/bj_bot_v1.pt")

    # Input: Your multi-card image
    input_image = "C:/Users/aferr/Desktop/bj_bot/tests/bjtest1.jpg"

    # Predict with lower confidence threshold and no NMS
    model.predict(source=input_image, save=True, conf=0.1, iou=0.5, name="multi_annotated", agnostic_nms=False)
    print(f"Annotated image saved in runs/detect/multi_annotated/")