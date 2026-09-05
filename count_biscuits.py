from ultralytics import YOLO
import cv2
import math

MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_detector-3\weights\best.pt"
VIDEO_PATH = r"testing_video.mp4"

CONFIDENCE = 0.30
# Maximum distance in pixels to consider a new tracker ID
# as the same physical biscuit.
MAX_MATCH_DISTANCE = 80
# How many frames an existing biscuit can disappear before
# we stop trying to match it.
MAX_MISSED_FRAMES = 15
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
print("Video opened successfully.")
# Our own permanent biscuit ID counter.
next_biscuit_id = 1
# Total biscuits counted.
total_biscuits = 0
biscuit_objects = {}
byte_to_biscuit = {}
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 +(y2 - y1) ** 2)
def find_existing_biscuit(center, current_frame, used_biscuit_ids):
    best_id = None
    best_distance = MAX_MATCH_DISTANCE
    for biscuit_id, data in biscuit_objects.items():
        # Do not allow one existing biscuit to be matched
        # to multiple detections in the same frame.
        if biscuit_id in used_biscuit_ids:
            continue
        last_seen = data["last_seen"]
        # Ignore objects that have been missing for too long.
        if current_frame - last_seen > MAX_MISSED_FRAMES:
            continue
        old_center = data["center"]
        distance = calculate_distance(old_center, center)
        if distance < best_distance:
            best_distance = distance
            best_id = biscuit_id
    return best_id

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    frame_number += 1
    results = model.track(
        frame,
        conf=CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    result = results[0]
    annotated_frame = frame.copy()
    current_detections = []
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()

        for track_id, confidence, box in zip(track_ids,confidences, xyxy):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center = (center_x, center_y)
            current_detections.append({
                "byte_id": track_id,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": center
            })
    used_biscuit_ids = set()
    for detection in current_detections:
        byte_id = detection["byte_id"]
        center = detection["center"]

        if byte_id in byte_to_biscuit:
            biscuit_id = byte_to_biscuit[byte_id]
            # Make sure this ID still exists.
            if biscuit_id in biscuit_objects:
                used_biscuit_ids.add(biscuit_id)
                biscuit_objects[biscuit_id]["center"] = center
                biscuit_objects[biscuit_id]["last_seen"] = frame_number
                biscuit_objects[biscuit_id]["byte_id"] = byte_id
                detection["biscuit_id"] = biscuit_id
                continue

        biscuit_id = find_existing_biscuit(center,frame_number,used_biscuit_ids)
        if biscuit_id is not None:
            byte_to_biscuit[byte_id] = biscuit_id
            used_biscuit_ids.add(biscuit_id)
            biscuit_objects[biscuit_id]["center"] = center
            biscuit_objects[biscuit_id]["last_seen"] = frame_number
            biscuit_objects[biscuit_id]["byte_id"] = byte_id
            detection["biscuit_id"] = biscuit_id
            print(f"Tracker ID changed: ByteTrack {byte_id} -> Biscuit {biscuit_id}")
        else:
            biscuit_id = next_biscuit_id
            next_biscuit_id += 1
            total_biscuits += 1
            # Save permanent biscuit information.
            biscuit_objects[biscuit_id] = {"center": center,"last_seen": frame_number,"byte_id": byte_id}
            # Connect ByteTrack ID to permanent ID.
            byte_to_biscuit[byte_id] = biscuit_id
            used_biscuit_ids.add(biscuit_id)
            detection["biscuit_id"] = biscuit_id
            print(f"NEW BISCUIT | Biscuit ID: {biscuit_id} | ByteTrack ID: {byte_id} | Confidence: {detection['confidence']:.2f} | Total: {total_biscuits}")
    old_biscuit_ids = []
    for biscuit_id, data in biscuit_objects.items():
        if frame_number - data["last_seen"] > MAX_MISSED_FRAMES:
            old_biscuit_ids.append(biscuit_id)
    for biscuit_id in old_biscuit_ids:
        del biscuit_objects[biscuit_id]
    for detection in current_detections:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        biscuit_id = detection["biscuit_id"]
        confidence = detection["confidence"]
        cv2.rectangle(annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2)
        cv2.circle(annotated_frame,
            (center_x, center_y),
            5,
            (0, 255, 0),
            -1)
        label = f"ID:{biscuit_id} {confidence:.2f}"
        cv2.putText(annotated_frame,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2)
    cv2.putText(annotated_frame,
        f"Total Biscuits: {total_biscuits}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2)

    cv2.putText(annotated_frame,
        f"Frame: {frame_number}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2)
    cv2.putText(annotated_frame,
        f"Active IDs: {len(current_detections)}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.imshow("Biscuit Counting",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Total unique biscuits: {total_biscuits}")
print("=" * 60)