from ultralytics import YOLO

model = YOLO("../src/models/bj_bot_v1.pt")
print("success")

input = "C:/Users/aferr/Desktop/bj_bot/tests/12.jpg"

model.predict(source= input, save= True, name = "output")