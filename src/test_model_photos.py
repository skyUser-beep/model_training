from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
import os

# CONFIGURATION
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_v2\weights\best.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640
# LOAD MODEL
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
print("Classes:", model.names)

# SELECT IMAGES
root = tk.Tk()
root.withdraw()
image_paths = filedialog.askopenfilenames(
    title="Select images to test",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("JPG files", "*.jpg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
)
if not image_paths:
    print("No images selected.")
    exit()
print()
print(f"{len(image_paths)} image(s) selected.")
print()
# PROCESS IMAGES
for image_path in image_paths:
    print("----------------------------------------")
    print("Image:", os.path.basename(image_path))
    # Run YOLO
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )
    result = results[0]
    # Original image
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Could not read image.")
        continue
    # Number of detections
    if result.boxes is not None:
        object_count = len(result.boxes)
    else:
        object_count = 0
    print("Detected biscuits:", object_count)
    # Draw detections
    for box in result.boxes:
        # Coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # Confidence
        confidence = float(box.conf[0])
        # Class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        # Draw box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        # Label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    # Status
    if object_count > 0:
        status = "BISCUITS PRESENT"
        status_color = (0, 255, 0)
    else:
        status = "CONVEYOR EMPTY"
        status_color = (0, 0, 255)
    # Status background
    cv2.rectangle(
        image,
        (15, 15),
        (500, 85),
        (30, 30, 30),
        -1
    )
    cv2.putText(
        image,
        status,
        (30, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )
    # Detection count
    cv2.putText(
        image,
        f"Detected: {object_count}",
        (15, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    # Display
    window_name = "YOLO Biscuit Detection"
    cv2.imshow(
        window_name,
        image
    )
    print("Press:")
    print("  N = next image")
    print("  Q = quit")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()
        if key == ord("n"):
            break
cv2.destroyAllWindows()
print()
print("Testing finished.")