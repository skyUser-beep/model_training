from ultralytics import YOLO
import cv2
# CONFIGURATION

MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_v2\weights\best.pt"
VIDEO_PATH = r"v1.mp4"
CONFIDENCE = 0.10

# LOAD MODEL

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
print("Video opened successfully.")

# VARIABLES
# IDs of biscuits that have already been counted
counted_ids = set()
total_biscuits = 0
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
    # GET TRACK IDs
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        for track_id in track_ids:
            # COUNT ONLY NEW BISCUITS
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                total_biscuits += 1
                print(
                    f"New biscuit detected | "
                    f"ID: {track_id} | "
                    f"Total: {total_biscuits}"
                )
    # DISPLAY TOTAL COUNT
    cv2.putText(
        annotated_frame,
        f"Total Biscuits: {total_biscuits}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    # Display frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    # SHOW VIDEO
    cv2.imshow(
        "Biscuit Counting",
        annotated_frame
    )
    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()

# FINAL RESULT
print()
print("=" * 50)
print("FINAL RESULT")
print("=" * 50)
print(f"Total biscuits detected: {total_biscuits}")
print("=" * 50)