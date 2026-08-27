import cv2 # video processing
import numpy as np # used for array,image calc,sorting etc
import os # creating folder
import random # shuffle selected frames , divide frames b/w train and validation
import math # for calculation purposes

# CONFIGURATION
VIDEO_PATHS = [
    "finetune.mp4",
]
# How many frames to sample from each video.
# The script chooses frames spread throughout the video.
FRAMES_PER_VIDEO = 100
# Output YOLO dataset
OUTPUT_DIR = "biscuit_dataset"
IMAGE_TRAIN_DIR = os.path.join(
    OUTPUT_DIR, "images", "train"
)
IMAGE_VAL_DIR = os.path.join(
    OUTPUT_DIR, "images", "val"
)
LABEL_TRAIN_DIR = os.path.join(
    OUTPUT_DIR, "labels", "train"
)
LABEL_VAL_DIR = os.path.join(
    OUTPUT_DIR, "labels", "val"
)
# Validation percentage
VAL_PERCENT = 0.20 # 20% images go to validation,80 to training
# Maximum candidates shown for correction
MAX_CANDIDATES = 80 # 80 boxes max to find region
# Candidate filtering
MIN_BOX_AREA_PERCENT = 0.0005
MAX_BOX_AREA_PERCENT = 0.08
# Biscuit aspect ratio is allowed to vary because of perspective.
MIN_ASPECT = 0.25
MAX_ASPECT = 4.5
# Candidate overlap suppression
NMS_THRESHOLD = 0.35 # Non-Maximum Suppression avoid multiple counting

# CREATE DIRECTORIES
for directory in [
    IMAGE_TRAIN_DIR,
    IMAGE_VAL_DIR,
    LABEL_TRAIN_DIR,
    LABEL_VAL_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# GLOBAL GUI STATE

current_boxes = [] # boxes accepted
original_candidates = [] # boxes generated automatically by OpenCV
drawing = False
start_point = None
current_mouse = (0, 0)
frame = None
display = None
# IOU
# Intersection Over Union.- IoU = intersection area / union area
def box_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    intersection = iw * ih
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)
    union = area1 + area2 - intersection
    if union <= 0:
        return 0
    return intersection / union

# NON-MAXIMUM SUPPRESSION - removes overlapping duplicates
def nms_boxes(boxes, scores):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        remaining = []
        for j in order[1:]:
            if box_iou(boxes[i],boxes[j]) < NMS_THRESHOLD:
                remaining.append(j)
        order = np.array(remaining,dtype=np.int32)
    return [boxes[i]for i in keep]

# RECTANGULARITY
# measures how much contour fills its bounding rectangle
def rectangularity(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rectangle_area = w * h
    if rectangle_area <= 0:
        return 0
    return area / rectangle_area

# AUTOMATIC BISCUIT PROPOSALS
#automatic annotation engine - find region that might be biscuits.
#Mask 1 → color
#Mask 2 → local brightness
#Mask 3 → adaptive threshold
#Mask 4 → edges
def generate_candidates(image):
    h, w = image.shape[:2]
    image_area = h * w
    # Slight blur reduces tiny biscuit texture/noise.
    blurred = cv2.GaussianBlur(image,(5, 5),0)
    # LAB color space
    lab = cv2.cvtColor(blurred,cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # HSV
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    masks = []
    # MASK 1
    # Warm biscuit color
    # Use percentiles rather than hardcoded color thresholds.
    a_low = np.percentile(A, 35)
    a_high = np.percentile(A, 98)
    b_low = np.percentile(B, 35)
    b_high = np.percentile(B, 98)
    mask1 = ((A >= a_low) &(A <= a_high) &(B >= b_low) &(B <= b_high)).astype(np.uint8) * 255
    masks.append(mask1)
    # MASK 2
    # Local brightness
    local_mean = cv2.GaussianBlur(L,(0, 0),15)
    difference = cv2.absdiff(L,local_mean)
    diff_threshold = np.percentile(difference,55)
    mask2 = (difference > diff_threshold).astype(np.uint8) * 255
    masks.append(mask2)
    # MASK 3
    # Adaptive brightness
    mask3 = cv2.adaptiveThreshold(L,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-3)
    masks.append(mask3)
    # MASK 4
    # Edge-based rectangular regions
    edges = cv2.Canny(L,40,120)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
    mask4 = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,edge_kernel,iterations=2)
    masks.append(mask4)
    # Process every mask
    candidate_boxes = []
    candidate_scores = []
    min_area = image_area * MIN_BOX_AREA_PERCENT
    max_area = image_area * MAX_BOX_AREA_PERCENT
    for mask in masks:
        # Morphological cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_small)
        clean = cv2.morphologyEx(clean,cv2.MORPH_CLOSE,kernel_small,iterations=2)
        # Contours
        contours, _ = cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 5 or bh <= 5:
                continue
            aspect = bw/ float(bh)
            if aspect < MIN_ASPECT:
                continue
            if aspect > MAX_ASPECT:
                continue
            rect = rectangularity(contour)
            # Biscuit-like score
            score = 0
            # Rectangularity
            score += rect * 40
            # Prefer moderate aspect ratios
            aspect_score = min(aspect,1.0 / aspect)
            score += aspect_score * 20
            # Area score
            normalized_area = (area / image_area)
            if normalized_area > 0.002:
                score += 20
            # Bounding box dimensions
            if bw > 15 and bh > 15:
                score += 10
            candidate_boxes.append((x, y, x + bw, y + bh))
            candidate_scores.append(score)
    # NMS- Non Maximum Suppression
    boxes = nms_boxes(candidate_boxes,candidate_scores)
    # Limit number of proposals
    if len(boxes) > MAX_CANDIDATES:
        # Sort by area
        boxes = sorted(boxes,key=lambda b:
                (b[2] - b[0]) *
                (b[3] - b[1]),
            reverse=True
        )
        boxes = boxes[:MAX_CANDIDATES]
    return boxes

# DRAW GUI
#creates the annotation screen
#Orange = automatic proposal
#Green = accepted biscuit
#Blue = currently drawing a new box
#A = accept proposals
#R = reject all
#S = skip
#ENTER = save
def draw_interface():
    global frame
    global current_boxes
    global original_candidates
    global current_mouse
    canvas = frame.copy()
    # Candidate boxes
    for box in original_candidates:

        if box in current_boxes:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            1
        )
    # Accepted boxes
    for i, box in enumerate(
        current_boxes
    ):
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
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )
    # Manual rectangle while drawing
    if drawing and start_point is not None:
        x1, y1 = start_point
        x2, y2 = current_mouse
        cv2.rectangle(canvas,(x1, y1),(x2, y2),(255, 0, 0),2)
    # Instructions
    cv2.rectangle(canvas, (0, 0),(canvas.shape[1], 75),(25, 25, 25),-1)
    cv2.putText(canvas,
        "A=accept proposals | R=reject all | S=skip | ENTER=save",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )
    cv2.putText(canvas,
        "Left-drag=add box | Right-click=remove nearest",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )
    cv2.putText(canvas,
        f"Boxes: {len(current_boxes)}",
        (canvas.shape[1] - 130, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )
    return canvas
# MOUSE CALLBACK
def mouse_callback(event,x,y,flags,param):
    global drawing
    global start_point
    global current_mouse
    global current_boxes
    current_mouse = (x,y)
    # Left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x,y)
    # Left button up
    elif (event ==cv2.EVENT_LBUTTONUP):
        if not drawing:
            return
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        # Normalize coordinates
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        width = right - left
        height = bottom - top
        # Ignore accidental tiny clicks
        if width < 10 or height < 10:
            return
        current_boxes.append((left,top,right,bottom))
        start_point = None
    # Right click = delete nearest box
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_boxes) == 0:
            return
        best_index = None
        best_distance = float("inf")
        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            distance = math.sqrt((x - cx) ** 2 +(y - cy) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        # Delete only if reasonably close
        if best_index is not None:
            x1, y1, x2, y2 = current_boxes[best_index]
            diagonal = math.sqrt((x2 - x1) ** 2 +(y2 - y1) ** 2)
            if best_distance < diagonal:
                current_boxes.pop(best_index)
# SAVE YOLO LABEL
def save_yolo_label(image_path,boxes):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    label_path = os.path.splitext(image_path)[0] + ".txt"
    # Convert image path to labels directory
    if IMAGE_TRAIN_DIR in image_path:
        label_path = image_path.replace(
            IMAGE_TRAIN_DIR,
            LABEL_TRAIN_DIR
        )
    elif IMAGE_VAL_DIR in image_path:
        label_path = image_path.replace(
            IMAGE_VAL_DIR,
            LABEL_VAL_DIR
        )
    label_path = os.path.splitext(
        label_path
    )[0] + ".txt"

    with open(label_path,"w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Clamp
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            nw = bw / w
            nh = bh / h
            # class 0 = biscuit
            f.write(f"0 {cx:.6f} {cy:.6f} "f"{nw:.6f} {nh:.6f}\n")
# EXTRACT FRAME INDICES
def get_frame_indices(video_path,count):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open:",video_path)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return []
    count = min(count,total)
    # Spread samples across video
    indices = np.linspace(
        0,
        total - 1,
        count,
        dtype=np.int32
    )
    return list(np.unique(indices))
# MAIN
cv2.namedWindow("Biscuit Labeling",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Biscuit Labeling",mouse_callback)
all_frames = []
print()
print("=" * 60)
print("SEMI-AUTOMATIC BISCUIT LABELING")
print("=" * 60)
print()
# Collect frames from videos
for video_id, video_path in enumerate(VIDEO_PATHS):
    indices = get_frame_indices(video_path, FRAMES_PER_VIDEO)
    for index in indices:
        all_frames.append(
            (video_path, int(index), video_id)
        )
print(f"Total candidate frames: "
f""f"{len(all_frames)}")
# Shuffle so training/validation frames are varied
random.shuffle(all_frames)
# Process frames
saved_count = 0
skipped_count = 0
for frame_index,(video_path,target_frame,video_id) in enumerate(all_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES,target_frame)
    ret, original = cap.read()
    cap.release()
    if not ret:
        continue
    # Keep video resolution
    frame = original.copy()
    # Generate automatic proposals
    print(f"\n[{frame_index + 1}/"f"{len(all_frames)}] "f"Generating proposals...")
    original_candidates = (generate_candidates(frame))
    current_boxes = (list(original_candidates))
    # GUI loop
    while True:
        display = draw_interface()
        cv2.imshow("Biscuit Labeling",display)
        key = cv2.waitKey(30) & 0xFF
        # A = accept all automatic boxes
        if key == ord("a"):
            current_boxes = (list(original_candidates))
        # R = reject all
        elif key == ord("r"):
            current_boxes = []
        # S = skip frame
        elif key == ord("s"):
            skipped_count += 1
            break
        # ENTER = save
        elif key in (13,10):
            if len(current_boxes) == 0:
                print("WARNING : No boxes. Press ENTER again to save empty label.")
            # Decide train/val
            if random.random() < VAL_PERCENT:
                image_dir = (IMAGE_VAL_DIR)
                label_dir = (LABEL_VAL_DIR)
            else:
                image_dir = (IMAGE_TRAIN_DIR)
                label_dir = (LABEL_TRAIN_DIR)
            filename = (f"video{video_id}_"f"frame{target_frame:08d}.jpg")
            image_path = os.path.join(image_dir,filename)
            label_path = os.path.join(label_dir,filename.replace(".jpg",".txt"))
            cv2.imwrite(image_path,frame)
            # Save YOLO labels
            h, w = frame.shape[:2]
            with open(label_path,"w") as f:
                for box in current_boxes:
                    x1, y1, x2, y2 = box
                    x1 = max(0,min(w - 1, x1))
                    x2 = max(0,min(w - 1, x2))
                    y1 = max(0,min(h - 1, y1))
                    y2 = max(0,min(h - 1, y2))
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2 ) / h
                    nw = bw / w
                    nh = bh / h
                    f.write(f"0 "f"{cx:.6f} "f"{cy:.6f} "f"{nw:.6f} "f"{nh:.6f}\n")
            saved_count +=1
            print(f"Saved: {filename} | "f"Boxes: "f"{len(current_boxes)}")
            break
        # ESC = quit
        elif key == 27:
            print()
            print("Stopping...")
            cv2.destroyAllWindows()
            print(f"Saved frames: "f"{saved_count}")
            print(f"Skipped frames: "f"{skipped_count}")
            raise SystemExit
cv2.destroyAllWindows()
# DATASET YAML
yaml_path = os.path.join(OUTPUT_DIR,"data.yaml")
# Use forward slashes for YAML
absolute_dataset = os.path.abspath(OUTPUT_DIR).replace("\\", "/")
with open(yaml_path,"w") as f:
    f.write(f"path: {absolute_dataset}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("\n")
    f.write("names:\n")
    f.write("  0: biscuit\n")
print()
print("=" * 60)
print("LABELING COMPLETE")
print("=" * 60)
print()
print(f"Saved frames : {saved_count}")
print(f"Skipped      : {skipped_count}")
print(f"Dataset      : {OUTPUT_DIR}")
print(f"YAML         : {yaml_path}")
print()
print("Next step:")
print( "Train YOLO using dataset/data.yaml")