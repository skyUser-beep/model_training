from ultralytics import YOLO

model = YOLO("../yolo26n.pt")
model.train(
    data="biscuit_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    name="biscuit_detector"
)