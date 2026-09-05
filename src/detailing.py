from ultralytics import YOLO
import cv2
import csv
import os
# CONFIGURATION
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_detector-3\weights\best.pt"
VIDEO_PATH = "testing_video.mp4"
CONFIDENCE = 0.30
# Horizontal line positions as percentage of video height
LINE1_POSITION = 0.40
LINE2_POSITION = 0.60
# Minimum observations before allowing an object to be counted
MIN_OBSERVATIONS = 3
# Maximum allowed downward movement before considering
# the movement suspicious.
MAX_BACKWARD_MOVEMENT = 50
# Dot smoothing
# Smaller = smoother
# Larger = more responsive
SMOOTHING_ALPHA = 0.25
# Dot radius
DOT_RADIUS = 7
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
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("\nVideo information:")
print(f"Width        : {width}")
print(f"Height       : {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")
# CALCULATE COUNTING LINES
line1_y = int(height * LINE1_POSITION)
line2_y = int(height * LINE2_POSITION)
# Line 1 must be above Line 2 because
# biscuits move upward through the video.
if line1_y >= line2_y:
    print( "ERROR: LINE 1 must be above LINE 2.")
    cap.release()
    exit(1)
print(f"\nLine 1 Y : {line1_y}")
print(f"Line 2 Y : {line2_y}")
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
    exit(1)# TRACKING DATA
# IDs that crossed Line 1
line1_crossed_ids = set()
# IDs that crossed Line 2
line2_crossed_ids = set()
# Final counted IDs
counted_ids = set()
# All IDs ever seen
all_ids = set()
# Previous Y position for each ID
previous_positions = {}
# Number of observations for each ID
observation_count = {}
# Last frame where ID was seen
last_seen = {}
# Direction/state of each object
object_state = {}
# PERSISTENT DOT DATA
# Smoothed position of every tracked biscuit
# IMPORTANT:
# This is outside the frame loop so it is NOT reset every frame.
smoothed_positions = {}
# Last known position of every biscuit
# Even if ByteTrack temporarily loses an object,
# its last dot position remains visible.
last_dot_positions = {}
# CSV FILE
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
    results = model.track(frame,
        conf=CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    result = results[0]
    # CREATE CLEAN FRAME
    # We use the original frame instead of result.plot()
    # so that YOLO boxes, IDs and confidence values
    # are NOT displayed.
    annotated_frame = frame.copy()
    # DRAW LINE 1
    cv2.line(annotated_frame,
        (0, line1_y),
        (width, line1_y),
        (0, 0, 255),
        3
    )
    # DRAW LINE 2
    cv2.line(annotated_frame,
        (0, line2_y),
        (width, line2_y),
        (0, 0, 255),
        3
    )
    # CURRENT IDs
    current_ids = set()
    # PROCESS DETECTIONS
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = (boxes.id.int().cpu() .tolist())
        xyxy = (boxes.xyxy.cpu().tolist())
        confidences = (boxes.conf.cpu().tolist())
        # PROCESS EACH BISCUIT
        for box, track_id, confidence in zip(xyxy,track_ids,confidences):
            x1, y1, x2, y2 = box
            # Biscuit center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            track_id = int(track_id)
            # TRACKING INFORMATION
            current_ids.add(track_id)
            all_ids.add(track_id)
            # OBSERVATION COUNT
            observation_count[track_id] = (observation_count.get(track_id, 0) + 1)
            # LAST SEEN
            last_seen[track_id] = frame_number
            # PREVIOUS POSITION
            previous_y = previous_positions.get(track_id,None)
            # INITIAL STATE
            if track_id not in object_state:
                object_state[track_id] = "BEFORE_LINE_2"
            state = object_state[track_id]
            crossed_line1 = False
            crossed_line2 = False
            # MOVEMENT CHECK
            moving_forward = True
            if previous_y is not None:
                movement = (center_y - previous_y)
                # Positive Y movement means downward.
                # Our biscuits are supposed to move upward.
                # If they suddenly move downward by a large
                # amount, treat that movement as suspicious.
                if movement > MAX_BACKWARD_MOVEMENT:
                    moving_forward = False
            # LINE 2 CROSSING
            # Biscuit moves upward.
            # Therefore:
            # Previous Y > Line 2
            # Current Y <= Line 2
            # Example:
            # Previous = 310
            # Line 2  = 288
            # Current  = 280
            # Biscuit crossed Line 2.
            if previous_y is not None and previous_y > line2_y >= center_y and moving_forward:
                if track_id not in line2_crossed_ids and observation_count[track_id] >= MIN_OBSERVATIONS :
                    line2_crossed_ids.add( track_id)
                    object_state[track_id] = "AFTER_LINE_2"
                    crossed_line2 = True
                    print(f"[LINE 2] "f"Frame={frame_number} "f"ID={track_id}")
            # LINE 1 CROSSING
            # Line 1 is above Line 2.
            # The biscuit must first cross Line 2.
            # Then it crosses Line 1.
            if previous_y is not None and previous_y > line1_y >= center_y and moving_forward:
                # Make sure Line 2 was crossed first.
                if track_id in line2_crossed_ids:
                    if track_id not in line1_crossed_ids:
                        line1_crossed_ids.add(track_id)
                        object_state[track_id] = "AFTER_LINE_1"
                        crossed_line1 = True
                        # FINAL COUNT
                        if  track_id not in counted_ids:
                            counted_ids.add(track_id)
                            print(f"[COUNTED] Frame={frame_number} ID={track_id} TOTAL={len(counted_ids)}")
            # UPDATE PREVIOUS POSITION
            previous_positions[track_id] = center_y
            # CURRENT STATE
            state = object_state[track_id]
            # CSV LOG
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
            # SMOOTH DOT POSITION
            if track_id not in smoothed_positions:
                # First time this biscuit appears.
                smoothed_positions[track_id] = (center_x,center_y)
            else:
                old_x, old_y = (smoothed_positions[track_id])
                # Smooth X
                smooth_x = ( SMOOTHING_ALPHA * center_x+ (1 - SMOOTHING_ALPHA) * old_x)
                # Smooth Y
                smooth_y = (SMOOTHING_ALPHA * center_y + (1 - SMOOTHING_ALPHA) * old_y)
                smoothed_positions[track_id] = ( int(smooth_x),int(smooth_y))
            # SAVE LAST KNOWN DOT POSITION
            last_dot_positions[track_id] = (smoothed_positions[track_id])
    # DRAW ALL PERSISTENT DOTS
    # IMPORTANT:
    # We draw EVERY previously detected biscuit here.
    # Therefore dots remain visible even when the biscuit
    # disappears from the current detection list.
    for saved_id, (dot_x, dot_y) in last_dot_positions.items():
        # GREEN = COUNTED
        # RED = NOT YET COUNTED
        if saved_id in counted_ids:
            dot_color = (0,255, 0)
        else:
            dot_color = (0,0,255)
        # DRAW DOT
        cv2.circle(annotated_frame,
            (dot_x, dot_y),
            DOT_RADIUS,
            dot_color,
            -1
        )
    # FINAL COUNT
    final_count = len(counted_ids)
    # COUNTER BACKGROUND
    cv2.rectangle(annotated_frame,
        (10, 10),
        (360, 75),
        (0, 0, 0),
        -1
    )
    # COUNTER TEXT
    cv2.putText(annotated_frame,
        f"TOTAL BISCUITS: {final_count}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    # STATISTICS
    current_count = len(current_ids)
    total_unique = len(all_ids)
    line1_count = len(line1_crossed_ids)
    line2_count = len(line2_crossed_ids)
    final_count = len(counted_ids)
    # SAVE OUTPUT VIDEO
    writer.write(annotated_frame)
    # DISPLAY
    cv2.imshow("Two-Line Biscuit Tracking",annotated_frame)

    # KEYBOARD
    key = cv2.waitKey(1) & 0xFF
    # Q = quit
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
    f.write( "=" * 60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write( f"Confidence: {CONFIDENCE}\n")
    f.write(f"Line 1 Y: {line1_y}\n")
    f.write(f"Line 2 Y: {line2_y}\n\n")
    f.write(f"Frames processed: {frame_number}\n")
    f.write( f"Unique IDs: {len(all_ids)}\n")
    f.write(f"Line 1 crossed:{len(line1_crossed_ids)}\n")
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
print(f"Line 1 crossed   : {len(line1_crossed_ids)}")
print(f"Line 2 crossed   : {len(line2_crossed_ids)}")
print(f"FINAL COUNT      : {len(counted_ids)}")
print()
print("Output files:")
print(f"Video  : {VIDEO_OUTPUT}")
print(f"CSV    : {CSV_OUTPUT}")
print(f"IDs    :{IDS_OUTPUT}")
print(f"Summary: {SUMMARY_OUTPUT}")
print("=" * 70)