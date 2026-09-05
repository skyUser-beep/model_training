from ultralytics import YOLO
import csv
import os
import cv2

MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_detector-3\weights\best.pt"
VIDEO_PATH = r"v1.mp4"

CONFIDENCE = 0.50

LINE1_POSITION = 0.65     # lower line
LINE2_POSITION = 0.40     # upper line

MIN_OBSERVATIONS=3
MAX_BACKWARD_MOVEMENT=50

OUTPUT_DIR="diagnostic_results"
os.makedirs(OUTPUT_DIR,exist_ok=True)

VIDEO_OUTPUT=os.path.join(OUTPUT_DIR,"video.mp4")
CSV_OUTPUT = os.path.join(OUTPUT_DIR,"horizontal_tracking_log.csv")
IDS_OUTPUT = os.path.join(OUTPUT_DIR,"horizontal_counted_ids.txt")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR,"horizontal_summary.txt")

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
print("Video opened successfully.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <=0:
    fps=25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)
print(f"Width       : {frame_width}")
print(f"Height      : {frame_height}")
print(f"FPS         : {fps:.2f}")
print(f"Total frames: {total_frames}")
print("=" * 60)

line1_y = int(frame_height * LINE1_POSITION)
line2_y = int(frame_height * LINE2_POSITION)

if line1_y <= line2_y:
    print()
    print("Error")
    print("For BOTTOM -> TOP Movement")
    print("LINE 1 must be below LINE 2")
    cap.release()
    exit(1)

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
writer=cv2.VideoWriter(VIDEO_OUTPUT,fourcc,fps,(frame_width,frame_height))

if not writer.isOpened():
    print("ERROR: Could not open video.")
    cap.release()
    exit(1)

print()
print("COUNTING DIRECTION")
print("=" * 60)
print(f"LINE 1 Y = {line1_y}  -> crossed FIRST")
print(f"LINE 2 Y = {line2_y}  -> crossed SECOND")
print("Direction = BOTTOM -> TOP")
print("Counting = LINE 1 -> LINE 2")
print("=" * 60)

# Track IDs that crossed LINE 1.
line1_crossed_ids = set()

# Track IDs that crossed LINE 2.
line2_crossed_ids = set()

# Track IDs that have already been counted.
counted_ids = set()
all_ids=set()

# Final biscuit count.
total_biscuits = 0

previous_positions = {}

track_observations = {}
last_seen={}
csv_file=open(CSV_OUTPUT,"w",newline="",encoding="utf-8")
csv_writer=csv.writer(csv_file)

csv_writer.writerow(["frame","track_id",
    "center_x","center_y",
    "confidence","state",
    "line1_crossed","line2_crossed",
    "counted"])

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
        iou=0.50,
        imgsz=1280,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    result = results[0]

    annotated_frame = frame.copy()

    cv2.line(annotated_frame,
        (0, line1_y),
        (frame_width, line1_y),
        (0, 255, 255),
        3)

    cv2.putText(annotated_frame,
        "LINE 1 - FIRST",
        (20, line1_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2)

    cv2.line(annotated_frame,
        (0, line2_y),
        (frame_width, line2_y),
        (0, 255, 0),
        3)

    cv2.putText(annotated_frame,
        "LINE 2 - SECOND",
        (20, line2_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2)
    detections=[]

    if  result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        for track_id, confidence, box in zip(track_ids,confidences,coordinates):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            track_id=int(track_id)
            detections.append({"track_id":track_id,
                               "x":center_x,"y":center_y,
                               "confidence": confidence,"x1":x1
                               ,"y1":y1,"x2":x2,"y2":y2})
        for detection in detections:
            track_id=detection['track_id']
            center_x=detection['x']
            center_y=detection['y']
            confidence=detection['confidence']
            x1=detection['x1']
            y1=detection['y1']
            x2=detection['x2']
            y2=detection['y2']

            all_ids.add(track_id)
            previous_y = previous_positions.get(track_id,None)

            track_observations[track_id]=track_observations.get(track_id,0)+1
            last_seen[track_id]=frame_number

            crossed_line1=False
            crossed_line2=False
            moving_forward=True

            if previous_y is not None :
                movement=center_y-previous_y

                if movement>MAX_BACKWARD_MOVEMENT:
                    moving_forward=False

            if previous_y is not None and previous_y > line1_y >= center_y and moving_forward:
                if track_id not in line1_crossed_ids and track_observations[track_id]>=MIN_OBSERVATIONS:
                    line1_crossed_ids.add(track_id)
                    crossed_line1 = True
                    print(f"[LINE 1 FIRST] Frame={frame_number} ID={track_id}")
            if previous_y is not None and previous_y < line2_y <= center_y and moving_forward:
                if track_id not in line2_crossed_ids:
                    line2_crossed_ids.add(track_id)
                    crossed_line2=True
                    print(
                        f"[LINE 2 SECOND] Frame={frame_number} ID={track_id}")

                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_biscuits+=1
                        print(f"==========================================")
                        print(f"BISCUIT COUNTED")
                        print(f"ID       : {track_id}")
                        print(f"Frame    : {frame_number}")
                        print(f"TOTAL    : {total_biscuits}")
                        print(f"==========================================")

            previous_positions[track_id] = center_y
            if track_id in counted_ids:
                box_color = (0, 255, 0)
            elif track_id in line1_crossed_ids:
                box_color = (0, 255, 255)
            else:
                box_color = (255, 255, 255)

            cv2.rectangle(annotated_frame,
                          (x1,y2),
                          (x2,y2),
                          box_color,
                          2)

            if track_id in counted_ids:
                dot_color = (0, 255, 0)
            elif track_id in line1_crossed_ids:
                dot_color = (0, 255, 255)
            else:
                dot_color = (255, 255, 255)

            cv2.circle(annotated_frame,
                          (center_x,center_y),
                            6,
                               dot_color,
                                 -1)
            if track_id in counted_ids:
                 id_text = f"ID:{track_id} COUNTED"
            elif track_id in line1_crossed_ids:
                id_text = f"ID:{track_id} L1"
            else:
                 id_text = f"ID:{track_id}"

            cv2.putText(annotated_frame,
                          id_text,(
                    x1,max(y1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2)
            conf_text = f"{confidence:.2f}"
            cv2.putText(annotated_frame,
                conf_text,
                (x1,min(y2 + 18,frame_height - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)
            if track_id in counted_ids:
                state="COUNTED"
            elif track_id in line1_crossed_ids:
               state="AFTER_LINE_1"
            else:
                state="BEFORE_LINE_1"

            csv_writer.writerow([
                frame_number,
                track_id,
                center_x,
                center_y,
                f"{confidence:.4f}",
                state,
                crossed_line1,
                crossed_line2,
                track_id in counted_ids
            ])
    active_tracks=len(detections)
    cv2.rectangle(annotated_frame,
                  (10,10),
                  (500,220),
                  (0,0,0),
                  -1)
    cv2.putText(annotated_frame,
        f"TOTAL BISCUITS: {total_biscuits}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3)
    cv2.putText(annotated_frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2)

    cv2.putText(annotated_frame,
        f"Active Tracks: {active_tracks}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2)

    cv2.putText(annotated_frame,
        f"Line 1 crossed: {len(line1_crossed_ids)}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2)

    cv2.putText(annotated_frame,
        f"Line 2 crossed: {len(line2_crossed_ids)}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2)
    writer.write(annotated_frame)

    cv2.imshow("Biscuit Counting - LINE 1 -> LINE 2",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

with open(IDS_OUTPUT,"w",encoding="utf-8") as f:
    for track_id in sorted(counted_ids):
        f.write( f"{track_id}\n")
with open(SUMMARY_OUTPUT, "w",encoding="utf-8") as f:
    f.write("HORIZONTAL TWO-LINE BISCUIT TRACKING\n")
    f.write( "=" * 70 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write( f"Confidence: {CONFIDENCE}\n")
    f.write("Direction: BOTTOM -> TOP\n")
    f.write("Counting: LINE 1 -> LINE 2\n\n")
    f.write(f"Line 1 Y: {line1_y}\n")
    f.write(f"Line 2 Y: {line2_y}\n\n")
    f.write( f"Frames processed: {frame_number}\n")
    f.write(f"Unique ByteTrack IDs: {len(all_ids)}\n")
    f.write(f"Line 1 crossed: {len(line1_crossed_ids)}\n")
    f.write(f"Line 2 crossed: {len(line2_crossed_ids)}\n")
    f.write( f"FINAL COUNT: {total_biscuits}\n\n")
    f.write("Counted IDs:\n")

    for track_id in sorted(counted_ids):
        observations=track_observations.get(track_id,0)

        f.write(f"{track_id}: {observations}\n")


print()
print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"LINE 1 crossed : {len(line1_crossed_ids)}")
print(f"LINE 2 crossed : {len(line2_crossed_ids)}")
print(f"TOTAL COUNT    : {total_biscuits}")
print("=" * 60)