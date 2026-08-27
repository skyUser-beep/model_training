from ultralytics import YOLO
# CONFIGURATION
MODEL_PATH = r"runs\detect\biscuit_detector\weights\best.pt"
DATASET = r"biscuit_dataset\data.yaml"
# Fine-tuning epochs
EPOCHS = 40
# Image resolution
IMAGE_SIZE = 640
# Small batch because you are using CPU
BATCH_SIZE = 4

# LOAD YOUR BEST EXISTING MODEL
model = YOLO(MODEL_PATH)
print("Loaded:")
print(MODEL_PATH)
# FINE-TUNE
results = model.train(
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    # CPU
    device="cpu",
    # Keep training reproducible
    seed=42,
    # DATA AUGMENTATION
    degrees=5,
    translate=0.05,
    scale=0.20,
    shear=2,
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.25,
    # MOSAIC
    mosaic=0.5,
    close_mosaic=10,
    # TRAINING
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=2,
    # VALIDATION
    val=True,
    plots=True,
    # SAVING
    project="runs/detect",
    name="biscuit_v2",
    exist_ok=True,
    save=True,
    save_period=10,
    # WORKERS
    workers=0,
    verbose=True
)

print()
print("====================================")
print("FINE-TUNING V2 FINISHED")
print("====================================")
print()
print("New Best model should be:")
print(r"runs\detect\biscuit_v2\weights\best.pt")