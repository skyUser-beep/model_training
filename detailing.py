from ultralytics import YOLO
import cv2
import csv
import os
# CONFIGURATION
# Put best.pt inside your project, for example:
# project/
# ├── best.pt
# ├── finetune.mp4
# └── two_line_tracking.py
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_v2\weights\best.pt"
VIDEO_PATH = "con.mp4"
CONFIDENCE = 0.50
# Line positions as percentage of video width
# Example:
# 0.40 = 40% of video width
# 0.60 = 60% of video width
LINE1_POSITION = 0.40
LINE2_POSITION = 0.60
# Minimum observations before allowing an object to be counted
# This helps reject very short/noisy detections.
MIN_OBSERVATIONS = 3
# Maximum allowed movement backwards in pixels before
# considering the object suspicious.
MAX_BACKWARD_MOVEMENT = 50
# Output directory
OUTPUT_DIR = "diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_OUTPUT = os.path.join(OUTPUT_DIR,"two_line_tracking_output.mp4")
CSV_OUTPUT = os.path.join(OUTPUT_DIR,"tracking_log.csv")
IDS_OUTPUT = os.path.join(OUTPUT_DIR,"counted_ids.txt")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR,"summary.txt")
# LOAD MODEL
print("=" * 70)
print("TWO-LINE BISCUIT TRACKING")
print("=" * 70)
print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)
print("\nVideo information:")
print(f"Width        : {width}")
print(f"Height       : {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")
# CALCULATE COUNTING LINES
line1_x = int(width * LINE1_POSITION)
line2_x = int(width * LINE2_POSITION)
# Make sure Line 1 is before Line 2
if line1_x >= line2_x:
    print("ERROR: LINE 1 must be before LINE 2.")
    cap.release()
    exit(1)
print(f"\nLine 1 X : {line1_x}")
print(f"Line 2 X : {line2_x}")
# VIDEO WRITER
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    fps,
    (width, height)
)
if not writer.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit(1)
# TRACKING DATA
# IDs that crossed Line 1
line1_crossed_ids = set()
# IDs that crossed Line 2
line2_crossed_ids = set()
# Final counted IDs
counted_ids = set()
# All IDs ever seen
all_ids = set()
# Previous X position
previous_positions = {}
# Number of observations for each ID
observation_count = {}
# Last frame where ID was seen
last_seen = {}
# Direction/state of each object
object_state = {}
# CSV
csv_file = open(
    CSV_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
)
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
# PROCESS VIDEO
frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    # YOLO TRACKING
    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    result = results[0]
    annotated_frame = result.plot()
    # DRAW LINE 1
    cv2.line(
        annotated_frame,
        (line1_x, 0),
        (line1_x, height),
        (255, 0, 0),
        3
    )
    cv2.putText(
        annotated_frame,
        "LINE 1",
        (line1_x + 10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )
    # DRAW LINE 2
    cv2.line(
        annotated_frame,
        (line2_x, 0),
        (line2_x, height),
        (0, 0, 255),
        3
    )
    cv2.putText(
        annotated_frame,
        "LINE 2",
        (line2_x + 10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    # CURRENT IDs
    current_ids = set()
    # PROCESS DETECTIONS
    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):
        boxes = result.boxes
        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )
        xyxy = (
            boxes.xyxy
            .cpu()
            .tolist()
        )
        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )
        for box, track_id, confidence in zip(
            xyxy,
            track_ids,
            confidences
        ):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            track_id = int(track_id)
            current_ids.add(track_id)
            all_ids.add(track_id)
            # OBSERVATION COUNT
            observation_count[track_id] = (observation_count.get(track_id, 0) + 1)
            last_seen[track_id] = frame_number
            # PREVIOUS POSITION
            previous_x = previous_positions.get(track_id,None)
            # INITIAL STATE
            if track_id not in object_state:
                object_state[track_id] = "BEFORE_LINE_1"
            state = object_state[track_id]
            crossed_line1 = False
            crossed_line2 = False
            # MOVEMENT CHECK
            moving_forward = True
            if previous_x is not None:
                movement = center_x - previous_x
                # If object suddenly moves far backwards,
                # don't allow it to advance its state.
                if movement < -MAX_BACKWARD_MOVEMENT:
                    moving_forward = False
            # LINE 1 CROSSING
            if  previous_x is not None and previous_x < line1_x <= center_x and moving_forward :
                if track_id not in line1_crossed_ids and observation_count[track_id] >= MIN_OBSERVATIONS:
                    line1_crossed_ids.add(track_id)
                    object_state[track_id] = "AFTER_LINE_1"
                    crossed_line1 = True
                    print(f"[LINE 1] Frame={frame_number} ID={track_id}")
            # LINE 2 CROSSING
            if  previous_x is not None and previous_x < line2_x <= center_x and moving_forward:
                if track_id in line1_crossed_ids:
                    if track_id not in line2_crossed_ids:
                        line2_crossed_ids.add(track_id)
                        object_state[track_id] = "AFTER_LINE_2"
                        crossed_line2 = True
                        # FINAL COUNT
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            print(f"[COUNTED] Frame={frame_number}  ID={track_id}  TOTAL={len(counted_ids)}")
            # UPDATE POSITION
            previous_positions[track_id] = center_x
            # CURRENT STATE
            state = object_state[track_id]
            # CSV
            csv_writer.writerow([
                frame_number,
                track_id,
                center_x,
                center_y,
                f"{confidence:.3f}",
                state,
                track_id in line1_crossed_ids,
                track_id in line2_crossed_ids,
                track_id in counted_ids
            ])
            # VISUAL DIAGNOSTIC
            if track_id in counted_ids:
                label = f"ID {track_id} COUNTED"
                text_color = (
                    0,
                    255,
                    0
                )
            elif track_id in line1_crossed_ids:
                label = f"ID {track_id} AFTER L1"
                text_color = (
                    0,
                    255,
                    255
                )
            else:
                label = f"ID {track_id} BEFORE L1"
                text_color = (
                    0,
                    165,
                    255
                )
            # Center point
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                6,
                text_color,
                -1
            )
            # ID label
            cv2.putText(
                annotated_frame,
                label,
                (
                    center_x - 70,
                    center_y - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2
            )
    # STATISTICS
    current_count = len(current_ids)
    total_unique = len(all_ids)
    line1_count = len(line1_crossed_ids)
    line2_count = len(line2_crossed_ids)
    final_count = len(counted_ids)
    # BACKGROUND PANEL
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (460, 170),
        (0, 0, 0),
        -1
    )
    # Current tracked
    cv2.putText(
        annotated_frame,
        f"Current tracked: {current_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    # Unique IDs
    cv2.putText(
        annotated_frame,
        f"Unique IDs: {total_unique}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    # Line 1
    cv2.putText(
        annotated_frame,
        f"Line 1 crossed: {line1_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    # Line 2
    cv2.putText(
        annotated_frame,
        f"Line 2 crossed: {line2_count}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    # Final count
    cv2.putText(
        annotated_frame,
        f"FINAL COUNT: {final_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )
    # FRAME NUMBER
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}/{total_frames}",
        (
            width - 250,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    # SAVE OUTPUT VIDEO
    writer.write(annotated_frame)
    # DISPLAY
    cv2.imshow("Two-Line Biscuit Tracking",annotated_frame)
    # Q = quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("\nStopped by user.")
        break
# CLEANUP
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()
# SAVE COUNTED IDS
with open(IDS_OUTPUT,"w",encoding="utf-8") as f:
    for track_id in sorted(counted_ids):
        f.write(f"{track_id}\n")
# SAVE SUMMARY
with open(SUMMARY_OUTPUT,"w",encoding="utf-8") as f:
    f.write("TWO-LINE BISCUIT TRACKING\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write(f"Confidence: {CONFIDENCE}\n")
    f.write(f"Line 1 X: {line1_x}\n")
    f.write(f"Line 2 X: {line2_x}\n\n")
    f.write(f"Frames processed: {frame_number}\n")
    f.write(f"Unique IDs: {len(all_ids)}\n")
    f.write( f"Line 1 crossed:{len(line1_crossed_ids)}\n")
    f.write(f"Line 2 crossed: {len(line2_crossed_ids)}\n")
    f.write(f"FINAL COUNT: {len(counted_ids)}\n\n")
    f.write("Counted IDs:\n")
    for track_id in sorted(counted_ids):
        observations = (observation_count.get(track_id,0))
        f.write(f"ID {track_id}: observations={observations}\n")
# FINAL RESULT
print()
print("=" * 70)
print("TWO-LINE TRACKING COMPLETE")
print("=" * 70)
print(f"Frames processed : {frame_number}")
print(f"Unique IDs       : {len(all_ids)}")
print(f"Line 1 crossed   :{len(line1_crossed_ids)}")
print(f"Line 2 crossed   : {len(line2_crossed_ids)}")
print(f"FINAL COUNT      : {len(counted_ids)}")
print()
print("Output files:")
print(f"Video  : {VIDEO_OUTPUT}")
print(f"CSV    : {CSV_OUTPUT}")
print(f"IDs    : {IDS_OUTPUT}")
print(f"Summary: {SUMMARY_OUTPUT}")
print("=" * 70)