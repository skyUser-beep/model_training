from ultralytics import YOLO
import cv2
import csv
import os


# CONFIGURATION
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_v3\weights\best.pt"
VIDEO_PATH = "finetune.mp4"

CONFIDENCE = 0.50

# Vertical lines as percentage of video width
LINE1_POSITION = 0.40
LINE2_POSITION = 0.60

# Direction of biscuit movement
MOVEMENT_DIRECTION = "LEFT TO RIGHT"

# Minimum number of observations before allowing a count
MIN_OBSERVATIONS = 3

# Maximum allowed backward movement in pixels
MAX_BACKWARD_MOVEMENT = 20

# Visual settings
DOT_RADIUS = 6
BOX_THICKNESS = 2
LINE_THICKNESS = 3

# Output
OUTPUT_DIR = "../diagnostic_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_OUTPUT = os.path.join(OUTPUT_DIR,"vertical_two_lines_tracking_output.mp4")
CSV_OUTPUT = os.path.join(OUTPUT_DIR,"vertical_tracking_log.csv")
IDS_OUTPUT = os.path.join(OUTPUT_DIR,"vertical_counted_ids.txt")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR,"vertical_summary.txt")


# START
print("=" * 70)
print("VERTICAL TWO-LINE BISCUIT TRACKING")
print("=" * 70)
print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video:")
    print(VIDEO_PATH)
    exit(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("\nVideo information:")
print(f"Width       : {width}")
print(f"Height      : {height}")
print(f"FPS         : {fps:.2f}")
print(f"Total frames: {total_frames}")

# CALCULATE VERTICAL LINES
line1_x = int(width * LINE1_POSITION)
line2_x = int(width * LINE2_POSITION)

if line1_x >= line2_x:
    print("\nERROR:")
    print("Line 1 must be LEFT of Line 2.")
    cap.release()
    exit(1)

print(f"\nLine 1 X    : {line1_x}")
print(f"Line 2 X    : {line2_x}")
print(f"Direction   : {MOVEMENT_DIRECTION}")

# VIDEO WRITER
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height))
if not writer.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit(1)

# TRACKING DATA
# IDs that crossed Line 1
line1_crossed_ids = set()
# IDs that crossed Line 2
line2_crossed_ids = set()
# IDs that completed both lines
counted_ids = set()
# Every ByteTrack ID ever seen
all_ids = set()
# Previous center X for every track
previous_x = {}
# Number of observations for every track
observation_count = {}
# Current state of every track
# LEFT TO RIGHT:
# BEFORE_LINE_1
# AFTER_LINE_1
# AFTER_LINE_2

object_state = {}

# CSV
csv_file = open(CSV_OUTPUT,"w",newline="",encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame",
    "track_id",
    "center_x",
    "center_y",
    "confidence",
    "state",
    "line1_crossed",
    "line2_crossed",
    "counted"
])

# FRAME LOOP
frame_number = 0
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # YOLO + BYTE TRACK
    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    result = results[0]
    # CREATE DISPLAY FRAME
    annotated_frame = frame.copy()
    # DRAW VERTICAL LINES
    cv2.line(annotated_frame,
        (line1_x, 0),
        (line1_x, height),
        (0, 0, 255),
        LINE_THICKNESS
    )
    cv2.line(annotated_frame,
        (line2_x, 0),
        (line2_x, height),
        (0, 0, 255),
        LINE_THICKNESS
    )
    # Line labels
    cv2.putText(annotated_frame,
        "LINE 1",
        (line1_x - 45, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    cv2.putText(annotated_frame,
        "LINE 2",
        (line2_x - 45, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    # GET DETECTIONS
    detections = []
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = (boxes.id.int().cpu().tolist())
        xyxy = (boxes.xyxy.cpu().tolist())
        confidences = (boxes.conf.cpu().tolist())
        for box, track_id, confidence in zip(xyxy,track_ids,confidences):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            track_id = int(track_id)
            detections.append({"track_id": track_id,
                "x": center_x,"y": center_y,
                "confidence": confidence,
                "x1": int(x1),"y1": int(y1),
                "x2": int(x2),"y2": int(y2)
            })
    # PROCESS EACH TRACK
    for detection in detections:
        track_id = detection["track_id"]
        center_x = detection["x"]
        center_y = detection["y"]
        confidence = detection["confidence"]
        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]
        all_ids.add(track_id)

        # Observation count
        observation_count[track_id] = (observation_count.get(track_id, 0) + 1)
        # First time seeing this ID
        if track_id not in object_state:
            if MOVEMENT_DIRECTION == "LEFT TO RIGHT":
                object_state[track_id] = "BEFORE_LINE_1"
            else:
                object_state[track_id] = "BEFORE_LINE_2"
        # Previous X
        previous = previous_x.get(track_id)
        # MOVEMENT CHECK
        moving_forward = True
        if previous is not None:
            movement = center_x - previous
            if MOVEMENT_DIRECTION == "LEFT TO RIGHT":
                if movement < -MAX_BACKWARD_MOVEMENT:
                    moving_forward = False
            else:
                if movement > MAX_BACKWARD_MOVEMENT:
                    moving_forward = False
        # LEFT TO RIGHT
        if MOVEMENT_DIRECTION == "LEFT TO RIGHT":
            # LINE 1
            if previous is not None and previous < line1_x <= center_x and moving_forward:
                if observation_count[track_id] >= MIN_OBSERVATIONS:
                    if track_id not in line1_crossed_ids:
                        line1_crossed_ids.add(track_id)
                        object_state[track_id] = "AFTER_LINE_1"
                        print(f"[LINE 1] Frame={frame_number} ID={track_id}")
            # LINE 2
            if previous is not None and previous < line2_x <= center_x and moving_forward:
                if track_id in line1_crossed_ids and track_id not in line2_crossed_ids:
                    line2_crossed_ids.add(track_id)
                    object_state[track_id] =  "AFTER_LINE_2"
                    print(f"[LINE 2] Frame={frame_number} ID={track_id}")

                    # COUNT
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        print(f"[COUNTED] Frame={frame_number} ID={track_id} TOTAL={len(counted_ids)}")

        # RIGHT TO LEFT
        else:
            # LINE 2 FIRST
            if previous is not None and previous > line2_x >= center_x and moving_forward:
                if observation_count[track_id]>= MIN_OBSERVATIONS:
                    if track_id not in line2_crossed_ids:
                        line2_crossed_ids.add(track_id)
                        object_state[track_id] = "AFTER_LINE_2"
                        print(f"[LINE 2] Frame={frame_number} ID={track_id}")

            # LINE 1 SECOND
            if previous is not None and previous > line1_x >= center_x and moving_forward:
                if track_id in line2_crossed_ids and track_id not in line1_crossed_ids:
                    line1_crossed_ids.add(track_id)
                    object_state[track_id] = "AFTER_LINE_1"
                    print(f"[LINE 1] Frame={frame_number} ID={track_id}")
                    # COUNT
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        print(
                            f"[COUNTED] Frame={frame_number} ID={track_id} TOTAL={len(counted_ids)}")

        # SAVE CURRENT X
        previous_x[track_id] = center_x
        # DRAW YOLO BOX
        if track_id in counted_ids:
            box_color = (0, 255, 0)
        else:
            box_color = (255, 255, 0)
        cv2.rectangle(annotated_frame,
            (x1, y1),
            (x2, y2),
            box_color,
            BOX_THICKNESS
        )
        # DRAW CENTER DOT
        cv2.circle(annotated_frame,
            (center_x, center_y),
            DOT_RADIUS,
            box_color,
            -1
        )
        # DRAW TRACK ID
        label = f"ID:{track_id} {confidence:.2f}"
        cv2.putText(annotated_frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2
        )
        # CSV LOG
        csv_writer.writerow([frame_number,
            track_id,
            center_x,center_y,
            f"{confidence:.4f}",
            object_state.get(
                track_id,
                "UNKNOWN"
            ),
            track_id in line1_crossed_ids,
            track_id in line2_crossed_ids,
            track_id in counted_ids
        ])

    # DISPLAY COUNT
    final_count = len(counted_ids)
    # Black information box
    cv2.rectangle(annotated_frame,
        (10, 10),
        (width - 10, 75),
        (0, 0, 0),
        -1
    )
    cv2.putText(annotated_frame,
        f"TOTAL BISCUITS: {final_count}",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    # DISPLAY TRACKING STATUS
    status_text = f"Line1: {len(line1_crossed_ids)} Line2: {len(line2_crossed_ids)} Tracks: {len(detections)}"
    cv2.putText(annotated_frame,
        status_text,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )
    # SAVE OUTPUT VIDEO
    writer.write(annotated_frame)
    # SHOW VIDEO
    cv2.imshow("Vertical Two-Line Biscuit Tracking",annotated_frame)
    # Play approximately at original video speed
    delay = max(1,int(1000 / fps))
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        print("\nStopped by user.")
        running = False

# CLEANUP
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

# SAVE COUNTED IDS
with open(IDS_OUTPUT, "w",encoding="utf-8") as f:
    for track_id in sorted(counted_ids):
        f.write(f"{track_id}\n")

# SAVE SUMMARY
with open(SUMMARY_OUTPUT,"w",encoding="utf-8") as f:
    f.write("VERTICAL TWO-LINE BISCUIT TRACKING\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write(f"Confidence: {CONFIDENCE}\n")
    f.write(f"Direction: {MOVEMENT_DIRECTION}\n")
    f.write(f"Line 1 X: {line1_x}\n")
    f.write(f"Line 2 X: {line2_x}\n")
    f.write(f"Frames Processed: {frame_number}\n")
    f.write(f"Unique ByteTrack IDs: {len(all_ids)}\n")
    f.write(f"Line 1 Crossed: {len(line1_crossed_ids)}\n")
    f.write(f"Line 2 Crossed: {len(line2_crossed_ids)}\n")
    f.write(f"Final Count: {len(counted_ids)}\n")
    f.write("\nCounted IDs:\n")
    for track_id in sorted(counted_ids):
        observations = observation_count.get(track_id,0)
        f.write(f"ID {track_id}: Observations={observations}\n")

# FINAL TERMINAL OUTPUT
print()
print("=" * 70)
print("VERTICAL TWO-LINE TRACKING COMPLETE")
print("=" * 70)
print(f"Frames Processed : {frame_number}")
print(f"Unique IDs       : {len(all_ids)}")
print(f"Line 1 crossed   : {len(line1_crossed_ids)}")
print(f"Line 2 crossed   : {len(line2_crossed_ids)}")
print(f"FINAL COUNT      : {len(counted_ids)}")
print()
print("Output Files")
print(f"Video   : {VIDEO_OUTPUT}")
print(f"CSV     : {CSV_OUTPUT}")
print(f"IDs     : {IDS_OUTPUT}")
print(f"Summary : {SUMMARY_OUTPUT}")
print("=" * 70)