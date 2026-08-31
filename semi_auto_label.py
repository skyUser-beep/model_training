import cv2
import os
import shutil
from ultralytics import YOLO
# CONFIGURATION
# Your existing trained YOLO model
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_v2\weights\best.pt"
# Existing extracted frames
RAW_FRAMES_DIR = r"dataset\raw_frames"
# Final YOLO dataset
OUTPUT_DIR = r"biscuit_dataset"
IMAGE_TRAIN_DIR = os.path.join(OUTPUT_DIR, "images", "train")
IMAGE_VAL_DIR = os.path.join(OUTPUT_DIR, "images", "val")
LABEL_TRAIN_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
LABEL_VAL_DIR = os.path.join(OUTPUT_DIR, "labels", "val")

# YOLO SETTINGS
# IMPORTANT:
# We use 0.20 because your current model was missing
# many biscuits at 0.50.
YOLO_CONFIDENCE = 0.20
# YOLO image size during prediction
YOLO_IMAGE_SIZE = 640
# CPU
YOLO_DEVICE = "cpu"
# Your model has:
# 0 = biscuit
BISCUIT_CLASS_ID = 0

# DATASET SPLIT
# 80% training
# 20% validation
TRAIN_PERCENT = 0.80

# CREATE DIRECTORIES
for directory in [
    IMAGE_TRAIN_DIR,
    IMAGE_VAL_DIR,
    LABEL_TRAIN_DIR,
    LABEL_VAL_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# GLOBAL GUI STATE
frame = None
current_boxes = []
original_candidates = []
drawing = False
start_point = None
current_mouse = (0, 0)

# LOAD YOLO MODEL
print()
print("=" * 65)
print("YOLO SEMI-AUTOMATIC BISCUIT LABELING")
print("=" * 65)
print()
if not os.path.exists(MODEL_PATH):
    print("ERROR: YOLO model not found:")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.isdir(RAW_FRAMES_DIR):
    print("ERROR: Raw frames folder not found:")
    print(RAW_FRAMES_DIR)
    raise SystemExit

print("Loading YOLO model...")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print()
print("Model loaded successfully.")
print("YOLO confidence:", YOLO_CONFIDENCE)
print("Prediction device:", YOLO_DEVICE)
print()

# GET RAW FRAME FILES
image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)
image_files = []
for filename in os.listdir(RAW_FRAMES_DIR):
    if filename.lower().endswith(image_extensions):
        full_path = os.path.join(RAW_FRAMES_DIR,filename)
        image_files.append(full_path)
# Sort frames naturally by filename
image_files.sort()
if len(image_files) == 0:
    print("ERROR: No images found in:")
    print(RAW_FRAMES_DIR)
    raise SystemExit
print(f"Found {len(image_files)} extracted frames.")
print()

# TRAIN / VALIDATION SPLIT
total_images = len(image_files)
train_count = int(total_images * TRAIN_PERCENT)
print("Dataset split:")
print(f"Training images   : {train_count}")
print(f"Validation images : "f"{total_images - train_count}")
print()

# IOU FUNCTION
def box_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    iw = max(0,inter_x2 - inter_x1)
    ih = max(0,inter_y2 - inter_y1)
    intersection = iw * ih
    area1 = (max(0, x2 - x1)*max(0, y2 - y1))
    area2 = (max(0, a2 - a1)*max(0, b2 - b1))
    union = (area1+area2-intersection)
    if union <= 0:
        return 0.0
    return intersection / union

# YOLO PREDICTION

def generate_yolo_candidates(image):
    candidates = []
    results = model.predict(
        source=image,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        device=YOLO_DEVICE,
        verbose=False
    )
    for result in results:
        if result.boxes is None:
            continue
        if len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, class_id, confidence in zip(
            boxes,
            classes,
            confidences
        ):
            class_id = int(class_id)
            # Only keep biscuit class
            if class_id != BISCUIT_CLASS_ID:
                continue
            x1, y1, x2, y2 = box
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            candidates.append(
                {
                    "box": (x1, y1, x2, y2),
                    "confidence": float(confidence)
                }
            )
    return candidates

# DRAW GUI
def draw_interface():
    global frame
    global current_boxes
    global original_candidates
    global current_mouse
    canvas = frame.copy()

    # YOLO candidate boxes
    # Orange
    for candidate in original_candidates:
        box = candidate["box"]
        confidence = candidate["confidence"]
        # Don't draw candidate if already accepted
        if box in current_boxes:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            2
        )
        cv2.putText(
            canvas,
            f"YOLO {confidence:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA
        )
    # Accepted boxes
    # Green
    for i, box in enumerate(current_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        cv2.putText(
            canvas,
            str(i + 1),
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Manual box being drawn
    # Blue
    if drawing and start_point is not None:
        x1, y1 = start_point
        x2, y2 = current_mouse
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Header
    header_height = 85
    cv2.rectangle(
        canvas,
        (0, 0),
        (
            canvas.shape[1],
            header_height
        ),
        (25, 25, 25),
        -1
    )
    cv2.putText(
        canvas,
        "A=Accept YOLO | R=Reject All | S=Skip | ENTER=Save | ESC=Quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "LEFT DRAG = Add Box | RIGHT CLICK = Delete Box",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"Accepted boxes: {len(current_boxes)}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return canvas

# MOUSE CALLBACK
def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    global drawing
    global start_point
    global current_mouse
    global current_boxes
    current_mouse = (x, y)

    # LEFT MOUSE BUTTON
    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    # LEFT MOUSE BUTTON RELEASE
    # Finish drawing
    elif event == cv2.EVENT_LBUTTONUP:
        if not drawing:
            return
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        width = right - left
        height = bottom - top
        # Ignore accidental tiny boxes
        if width < 10 or height < 10:
            start_point = None
            return
        new_box = (
            left,
            top,
            right,
            bottom
        )
        # Prevent exact duplicate box
        if new_box not in current_boxes:
            current_boxes.append(new_box)
        start_point = None

    # RIGHT CLICK
    # Delete nearest accepted box
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_boxes) == 0:
            return
        best_index = None
        best_distance = float("inf")
        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            distance = ((x - cx) ** 2+(y - cy) ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_index = i
        if best_index is not None:
            x1, y1, x2, y2 = (current_boxes[best_index])
            diagonal = ((x2 - x1) ** 2+(y2 - y1) ** 2) ** 0.5
            if best_distance < diagonal:
                current_boxes.pop(best_index)


# SAVE YOLO LABEL
def save_yolo_label(label_path,boxes,image_width,image_height):
    with open(label_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Clamp coordinates
            x1 = max(0,min(image_width - 1, x1))
            x2 = max(0,min(image_width - 1,x2))
            y1 = max(0, min( image_height - 1,y1))
            y2 = max(0,min(image_height - 1,y2))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            # YOLO normalized format
            cx = ((x1 + x2) / 2.0) / image_width
            cy = ((y1 + y2) / 2.0) / image_height
            nw = bw / image_width
            nh = bh / image_height
            # Class 0 = biscuit
            f.write(
                f"0 "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{nw:.6f} "
                f"{nh:.6f}\n"
            )

# CREATE DATASET YAML
def create_data_yaml():
    yaml_path = os.path.join(
        OUTPUT_DIR,
        "data.yaml"
    )
    absolute_dataset = os.path.abspath(
        OUTPUT_DIR
    ).replace("\\", "/")
    with open(yaml_path,"w") as f:
        f.write(f"path: {absolute_dataset}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: biscuit\n")
    return yaml_path

# MAIN GUI
cv2.namedWindow("Biscuit Labeling",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Biscuit Labeling",mouse_callback)
saved_count = 0
skipped_count = 0
print()
print("=" * 65)
print("STARTING LABELING")
print("=" * 65)
print()
print("Instructions:")
print()
print("A       = Accept all YOLO proposals")
print("R       = Reject all proposals")
print("S       = Skip this frame")
print("ENTER   = Save corrected labels")
print("ESC     = Quit")
print()
print("LEFT DRAG  = Add missing biscuit")
print("RIGHT CLICK = Delete incorrect biscuit")
print()
print("IMPORTANT:")
print("Check EVERY biscuit before pressing ENTER.")
print()

# PROCESS EACH IMAGE
for frame_index, image_path in enumerate(image_files):
    filename = os.path.basename(image_path)
    print()
    print("=" * 65)
    print(f"Frame {frame_index + 1}/"f"{total_images}")
    print(f"File: {filename}")
    # Read image
    original = cv2.imread(image_path)
    if original is None:
        print("WARNING: Could not read:")
        print(image_path)
        continue
    frame = original.copy()

    # YOLO prediction
    print("Running YOLO prediction...")
    candidate_data = (generate_yolo_candidates(frame))
    original_candidates = (candidate_data)
    # Start with YOLO boxes
    current_boxes = [candidate["box"]
        for candidate in candidate_data]
    print(f"YOLO proposals: "f"{len(current_boxes)}")
    print()
    print("Check the frame carefully.")
    print("Add missing biscuits manually.")
    print("Delete incorrect boxes.")
    # GUI LOOP
    while True:
        display = draw_interface()
        cv2.imshow("Biscuit Labeling",display)
        key = ( cv2.waitKey(30)& 0xFF)
        # A = ACCEPT YOLO PROPOSALS

        if key == ord("a"):
            current_boxes = [candidate["box"]
                for candidate in original_candidates]
            print( "Accepted all YOLO proposals.")
        # R = REJECT ALL
        elif key == ord("r"):
            current_boxes = []
            print("All boxes removed.")
        # S = SKIP
        elif key == ord("s"):
            skipped_count += 1
            print( "Skipped frame.")
            break
        # ENTER = SAVE
        elif key in (13, 10):
            if len(current_boxes) == 0:
                print()
                print("WARNING:")
                print( "There are ZERO boxes.")
                print("If this frame really contains no biscuits,")
                print("press ENTER again." )
                # Wait for another key
                confirmation = (cv2.waitKey(0)&0xFF)
                if confirmation not in (13,10):
                    continue
            # Decide train or validation
            if frame_index < train_count:
                image_dir = IMAGE_TRAIN_DIR
                label_dir = LABEL_TRAIN_DIR
                split_name = "TRAIN"
            else:
                image_dir = IMAGE_VAL_DIR
                label_dir = LABEL_VAL_DIR
                split_name = "VAL"
            # Create filename
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.jpg"
            image_output_path = (os.path.join(image_dir,output_filename))
            label_output_path = (os.path.join(label_dir, base_name + ".txt" ))
            # Save image
            cv2.imwrite(image_output_path, frame)
            # Save YOLO labels
            h, w = frame.shape[:2]
            save_yolo_label(label_output_path,current_boxes, w,h)
            saved_count += 1
            print()
            print(f"SAVED [{split_name}]")
            print(f"Image: "f"{image_output_path}")
            print(f"Boxes: "f"{len(current_boxes)}")
            break
        # ESC = QUIT
        elif key == 27:
            print()
            print("Stopping labeling...")
            cv2.destroyAllWindows()
            print()
            print(f"Saved frames   : "f"{saved_count}")
            print(f"Skipped frames : "f"{skipped_count}")
            print()
            raise SystemExit

# FINISH
cv2.destroyAllWindows()
# CREATE DATA.YAML
yaml_path = create_data_yaml()
# FINAL SUMMARY
print()
print("=" * 65)
print("LABELING COMPLETE")
print("=" * 65)
print()
print(f"Total source frames : "f"{total_images}")
print(f"Saved frames        : "f"{saved_count}")
print(f"Skipped frames      : "f"{skipped_count}")
print()
print("Dataset:")
print(os.path.abspath(OUTPUT_DIR))
print()
print("data.yaml:")
print(yaml_path)
print()
print("=" * 65)
print("NEXT STEP")
print("=" * 65)
print()
print("Run your finetune.py")
print()
print(r"New model will be:")
print(r"runs\detect\biscuit_v2\weights\best.pt")
print()