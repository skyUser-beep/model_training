from ultralytics import YOLO
import cv2
# CONFIGURATION
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_detector-3\weights\best.pt"
VIDEO_PATH = r"finetune.mp4"
CONFIDENCE = 0.50
# Counting line position
# Change this value if necessary
LINE_POSITION = 0.50
# LOAD MODEL
print("=" * 60)
print("BISCUIT COUNTING - COUNTING LINE TEST")
print("=" * 60)
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video width  : {width}")
print(f"Video height : {height}")
print(f"Video FPS    : {fps:.2f}")

# COUNTING LINE
line_x = int(width * LINE_POSITION)
print(f"Counting line Y position: {line_x}")
# VARIABLES
total_biscuits = 0
# IDs that have already crossed the line
counted_ids = set()
# Previous center position of each tracked biscuit
previous_positions = {}
frame_number = 0
# PROCESS VIDEO
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
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
    # DRAW DETECTIONS
    annotated_frame = result.plot()
    # DRAW COUNTING LINE
    cv2.line(annotated_frame,
        (line_x, 0),
        (line_x, height),
        (0, 0, 255),
        3
    )
    cv2.putText(annotated_frame,
        "COUNTING LINE",
        (line_x+10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    # GET TRACKED OBJECTS
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist()
        # Bounding box coordinates
        xyxy = boxes.xyxy.cpu().tolist()
        for box, track_id in zip(xyxy, track_ids):
            x1, y1, x2, y2 = box
            # Center of biscuit
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # DRAW CENTER POINT
            cv2.circle(annotated_frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )
            # Display tracking ID
            cv2.putText(annotated_frame,
                f"ID: {track_id}",
                (center_x - 30, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            # CHECK PREVIOUS POSITION
            if track_id in previous_positions:
                previous_x = previous_positions[track_id]
                # COUNT WHEN MOVING DOWN THROUGH LINE
                crossed_line =  previous_x < line_x <= center_x
                if crossed_line:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_biscuits += 1
                        print(f"Biscuit counted | ID: {track_id} | Total: {total_biscuits}")
            # Save current position
            previous_positions[track_id] = center_x
    # DISPLAY TOTAL
    cv2.rectangle(annotated_frame,
        (10, 10),
        (300, 70),
        (0, 0, 0),
        -1
    )
    cv2.putText(annotated_frame,
        f"Total Biscuits: {total_biscuits}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # DISPLAY VIDEO
    cv2.imshow("Biscuit Counting - Sideways",annotated_frame)
    # Q = QUIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
# FINAL RESULT
print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Confidence threshold : {CONFIDENCE}")
print(f"Total biscuits       : {total_biscuits}")
print("=" * 60)