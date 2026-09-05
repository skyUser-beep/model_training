from ultralytics import YOLO
import cv2
import csv
import os
import statistics

from detailing import IDS_OUTPUT

MODEL_PATH=""
VIDEO_PATH=""

CONFIDENCE=0.50
LINE1_POSITION=0.40
LINE2_POSITION=0.60

MOVEMENT_DIRECTION="BOTTOM TO TOP"
MIN_OBSERVATIONS=3
MAX_BACKWARD_MOVEMENT=50

STANDARD_WIDTH_PX=34.0
STANDARD_HEIGHT_PX=58.0

SIZE_TOLERANCE=0.05

MIN_WIDTH_PX= STANDARD_WIDTH_PX*(1-SIZE_TOLERANCE)
MAX_WIDTH_PX=STANDARD_WIDTH_PX*(1+SIZE_TOLERANCE)

MIN_HEIGHT_PX=STANDARD_HEIGHT_PX*(1-SIZE_TOLERANCE)
MAX_HEIGHT_PX=STANDARD_HEIGHT_PX*(1+SIZE_TOLERANCE)

MAX_SIZE_MEASUREMENTS=30

# VISUAL_MATCH_DISTANCE=50
# VISUAL_MAX_X_DISTANCE=35
# VISUAL_MAX_Y_DISTANCE=45
# VISUAL_MAX_MISSED_FRAMES=12

BOX_THICKNESS=2


SMOOTHING_ALPHA=0.35
VELOCITY_ALPHA=0.40

DOT_RADIUS=7

OUTPUT_DIR="diagnostic_results"
os.makedirs(OUTPUT_DIR,exist_ok=True)

VIDEO_OUTPUT=os.path.join(OUTPUT_DIR,"horizontal_size_test_output.mp4")
CSV_OUTPUT=os.path.join(OUTPUT_DIR,"horizontal_size_test_log.csv")
SUMMARY_OUTPUT=os.path.join(OUTPUT_DIR,"horizontal_size_summary.txt")
IDS=os.path.join(OUTPUT_DIR,"horizontal_size_counted_ids.txt")

# def distance(x1,y1,x2,y2):
#     return math.sqrt((x1-x2)**2+(y1-y2)**2)
#
# def clamp(value,min_value,max_value):
#     return max(min_value,min(value,max_value))
# def create_visual_object(visual_id,track_id,center_x,center_y,frame_number):
#     return {"track_id": track_id,"x": float(center_x),"y": float(center_y),
#             "prev_x":float(center_x),"prev_y":float(center_y),
#             "vx":0.0, "vy":0.0, "last_seen_frame": frame_number, "counted": False, "observations":1}

def get_size_status(width_px,height_px):
    width_ok= MIN_WIDTH_PX<= width_px<=MAX_WIDTH_PX
    height_ok=MIN_HEIGHT_PX<=height_px<MAX_HEIGHT_PX
    if width_ok and height_ok:
        return "ACCEPTED"
    return "REJECTED"

def check_size(width_px,height_px):
    width_ok=MIN_WIDTH_PX<=width_px<=MAX_WIDTH_PX
    height_ok=MIN_HEIGHT_PX<=height_px<=MAX_HEIGHT_PX
    return width_ok and height_ok

print("=" * 70)
print("HORIZONTAL BISCUIT SIZE + TWO-LINE TRACKING TEST")
print("=" * 70)

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

cap=cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening the Video")
    exit(1)

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps=cap.get(cv2.CAP_PROP_FPS)
if fps<=0:
    fps=25
total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information:")
print(f"Width        : {width}")
print(f"Height       : {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")

line1_y=int(height * LINE1_POSITION)
line2_y=int(height*LINE2_POSITION)

if line1_y>=line2_y:
    print("Error: Line 1 must be above Line2.")
    cap.release()
    exit(1)

print(f"\nLine 1 Y : {line1_y}")
print(f"Line 2 Y : {line2_y}")
print(f"Direction: {MOVEMENT_DIRECTION}")


print("\n SIZE TEST")
print(f"Standard width : {STANDARD_WIDTH_PX:.1f} px")
print(f"Standard height : {STANDARD_HEIGHT_PX:.1f} px")
print(f"Tolerance :  ±{SIZE_TOLERANCE*100:.0f}%")
print(f"Accepted width : {MIN_WIDTH_PX:.1f}-{MAX_WIDTH_PX:.1f}px")
print(f"Accepted height : {MIN_HEIGHT_PX:.1f}-{MAX_HEIGHT_PX:.1f} px")

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
writer=cv2.VideoWriter(VIDEO_OUTPUT,fourcc,fps,(width,height))

if not writer.isOpened():
    print("Error: Could not create output video.")
    cap.release()
    exit(1)

line1_crossed_ids=set()
line2_crossed_ids=set()
counted_ids=set()
all_ids=set()
prev_y={}
# prev_pos={}
observation_count={}
object_state={}

size_width={}
size_height={}


accepted_ids=set()
rejected_ids=set()
final_size_measurements={}

# visual_objects={}
#
# next_visual_id=1

csv_file=open(CSV_OUTPUT,"w",newline='',encoding="utf-8")
csv_writer=csv.writer(csv_file)
csv_writer.writerow(["frame","track_id"
                     ,"center_x","center_y","width_px","height_px","size_status",
                     "confidence","line1_crossed","line2_crossed","counted"])

frame_number=0
running=True

while running:
    ret,frame=cap.read()
    if not ret:
        break
    frame_number+=1

    results=model.track(frame,conf=CONFIDENCE,persist=True,tracker="bytetrack.yaml",verbose=False)
    result=results[0]
    annotated_frame=frame.copy()

    cv2.line(annotated_frame,
             (0,line1_y),
             (width,line1_y),
             (0,0,255),
             3)
    cv2.line(annotated_frame,
             (0,line2_y),
             (width,line2_y),
             (0,0,255),
             3)

    cv2.line(annotated_frame,
             (0,line2_y),
             (width,line2_y),
             (0,0,255),
             3)

    detections=[]
    if result.boxes is not None and result.boxes.id is not None:
        boxes=result.boxes
        track_ids=boxes.id.int().cpu().tolist()
        xyxy=boxes.xyxy.int().cpu().tolist()
        confidences=boxes.conf.cpu().tolist()

        for box,track_id,confidence in zip(xyxy,track_ids,confidences):
            x1,y1,x2,y2=box
            center_x=int((x1+x2)/2)
            center_y=int((y1+y2)/2)
            box_width=x2-x1
            box_height=y2-y1
            track_id=int(track_id)
            detections.append({"track_id": track_id,"x":center_x,"y":center_y,
                               "width":box_width,"height":box_height,
                               "confidence":confidence,"x1":x1,"y1":y1,"x2":x2,"y2":y2})

    for detection in detections:
        track_id=detection["track_id"]
        center_x=detection["x"]
        center_y=detection["y"]
        box_width=detection["width"]
        box_height=detection["height"]
        confidence=detection["confidence"]
        x1=detection["x1"]
        y1=detection["y1"]
        x2=detection["x2"]
        y2=detection["y2"]
        all_ids.add(track_id)

        observation_count[track_id]=observation_count.get(track_id,0)+1

        if track_id not in object_state:
            object_state[track_id]="BEFORE_LINE_2"

        if track_id not in size_width:
            size_width[track_id]=[]
            size_height[track_id]=[]

        size_width[track_id].append(box_width)
        size_height[track_id].append(box_height)


        if len(size_width[track_id])>MAX_SIZE_MEASUREMENTS:
            size_width[track_id].pop(0)

        if len(size_height[track_id])>MAX_SIZE_MEASUREMENTS:
            size_height[track_id].pop(0)

        median_width=statistics.median(size_width[track_id])
        median_height=statistics.median(size_height[track_id])

        current_size_status=get_size_status(median_width,median_height)

        prev=prev_y.get(track_id)

        moving_forward=True

        if prev is not None:
            movement=center_y-prev

            if movement> MAX_BACKWARD_MOVEMENT:
                moving_forward=False


        crossed_line2=False
        if prev is not None and prev > line2_y >= center_y and moving_forward:
            if track_id not in line1_crossed_ids and observation_count[track_id]>=MIN_OBSERVATIONS:
                line2_crossed_ids.add(track_id)
                object_state[track_id]="AFTER_LINE_2"
                crossed_line2=True
                print(f"[Line 2] Frame={frame_number},ID={track_id}, Size={median_width:.1f}x {median_height:.1f} Status={current_size_status}")

        crossed_line1=False
        if prev is not None and prev < line1_y <= center_y and moving_forward:
            if track_id in line2_crossed_ids and track_id not in line1_crossed_ids:
                line1_crossed_ids.add(track_id)
                object_state[track_id]="AFTER_LINE_1"
                crossed_line1=True
                final_width=statistics.median(size_width[track_id])
                final_height=statistics.median(size_height[track_id])
                final_size_measurements[track_id]={"width":final_width,"height":final_height}

                final_size_ok=check_size(final_width,final_height)

                if final_size_ok:
                    accepted_ids.add(track_id)
                    rejected_ids.discard(track_id)
                    final_size_status="ACCEPTED"
                else:
                    rejected_ids.add(track_id)
                    accepted_ids.discard(track_id)
                    final_size_status="REJECTED"

                print(f"[LINE 1] Frame= {frame_number} ID = {track_id} Final Size = {final_width:.1f}x {final_height:.1f} {final_size_status}")

                if final_size_ok and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    print(f"[COUNTED] Frame={frame_number} ID = {track_id} SIZE= {final_width:.1f}x {final_height:.1f} TOTAL= {len(counted_ids)}")
                elif not final_size_ok:
                    print(f"[REJECTED SIZE Frame={frame_number} ID={track_id} SIZE= {final_width:.1f}x {final_height:.1f}")


        prev_y[track_id]=center_y
        if track_id in counted_ids:
            box_color=0,255,0
        elif track_id in rejected_ids:
            box_color=0,0,255
        elif current_size_status=="ACCEPTED":
            box_color=255,255,0
        else:
            box_color=0,165,255


        cv2.rectangle(annotated_frame,
                      (x1,y2),
                      (x2,y2),
                      box_color,
                      3)
        cv2.circle(annotated_frame,
                   (center_x,center_y),
                   DOT_RADIUS,
                   box_color,
                   -1)
        id_text=f"ID:{track_id}"

        csv_writer.writerow([frame_number,track_id,center_x,center_y,f"{box_width:.2f},"
                                                                     f"{box_height:.2f},"
                                                                     f"{median_width:.2f},"
                                                                     f"{median_height:.2f}",
                                                                     current_size_status, f"{confidence:.4f}",
                             object_state.get(track_id,"UNKNOWN"),
                             track_id in line1_crossed_ids,
                             track_id in line2_crossed_ids,
                             track_id in counted_ids])
        final_count=len(counted_ids)
        cv2.rectangle(annotated_frame,
                      (10,10),
                      (460,80),
                      (0,0,0),
                      -1)
        cv2.putText(annotated_frame,
                    f"TOTAL BISCUITS: {final_count}",
                    (30,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (0,255,0),
                    2)

        status_text=f"Line2 : {len(line2_crossed_ids)} Line1 : {len(line1_crossed_ids)} Tracks: {len(detections)}"
        cv2.putText(annotated_frame,
                    status_text,
                    (20,height-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (255,255,255),
                    2)

        writer.write(annotated_frame)
        cv2.imshow("Horizontal Biscuit Size Tracking",annotated_frame)

        delay=max(1,int(1000/fps))
        key=(cv2.waitKey(delay) & 0xFF)
        if key== ord('q'):
            print("\n Stopped by user")
            running=False

cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

with open(IDS_OUTPUT, "w",encoding="utf-8") as f:
    for track_id in sorted(counted_ids):
        f.write(f"{track_id}\n")

# SAVE SUMMARY

with open(SUMMARY_OUTPUT,"w",encoding="utf-8") as f:
    f.write("HORIZONTAL BISCUIT SIZE + ""TWO-LINE TRACKING\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Video: {VIDEO_PATH}\n")
    f.write(f"Confidence: {CONFIDENCE}\n")
    f.write(f"Direction: {MOVEMENT_DIRECTION}\n")
    f.write(f"Line 1 Y: {line1_y}\n")
    f.write(f"Line 2 Y: {line2_y}\n")
    f.write(f"Standard width: {STANDARD_WIDTH_PX:.1f}\n")
    f.write(f"Standard height: {STANDARD_HEIGHT_PX:.1f}\n")
    f.write(f"Size tolerance: ±{SIZE_TOLERANCE * 100:.0f}%\n")
    f.write(f"Accepted width: {MIN_WIDTH_PX:.1f} - {MAX_WIDTH_PX:.1f}\n")
    f.write(f"Accepted height: {MIN_HEIGHT_PX:.1f} - {MAX_HEIGHT_PX:.1f}\n")
    f.write("\n")
    f.write(f"Frames processed: {frame_number}\n")
    f.write(f"Unique ByteTrack IDs: {len(all_ids)}\n")
    f.write(f"Line 2 crossed: {len(line2_crossed_ids)}\n")
    f.write(f"Line 1 crossed: {len(line1_crossed_ids)}\n")
    f.write(f"Accepted size IDs: {len(accepted_ids)}\n")
    f.write(f"Rejected size IDs: {len(rejected_ids)}\n")
    f.write(f"FINAL COUNT: {len(counted_ids)}\n\n")


    # COUNTED IDS
    f.write("COUNTED IDS:\n")
    for track_id in sorted(counted_ids):
        measurement = (final_size_measurements.get( track_id))
        if measurement is not None:
            f.write(f"ID {track_id}:{measurement['width']:.2f} x {measurement['height']:.2f} px\n")
        else:
            f.write(f"ID {track_id}\n")

    # REJECTED IDS
    f.write("\nREJECTED SIZE IDS:\n")
    for track_id in sorted(rejected_ids):
        measurement = (final_size_measurements.get(track_id))
        if measurement is not None:
            f.write(f"ID {track_id}: {measurement['width']:.2f} x{measurement['height']:.2f} px\n")
        else:
            f.write(f"ID {track_id}\n")

# FINAL TERMINAL OUTPUT

print()
print("=" * 70)
print("HORIZONTAL SIZE + TWO-LINE TRACKING COMPLETE")
print("=" * 70)
print(f"Frames processed :{frame_number}")
print(f"Unique IDs       :{len(all_ids)}")
print(f"Line 2 crossed   :{len(line2_crossed_ids)}")
print(f"Line 1 crossed   :{len(line1_crossed_ids)}")
print(f"Accepted sizes   :{len(accepted_ids)}")
print(f"Rejected sizes   :{len(rejected_ids)}")
print(f"FINAL COUNT      :{len(counted_ids)}")
print()
print("Output files:")
print(f"Video   : {VIDEO_OUTPUT}")
print(f"CSV     : {CSV_OUTPUT}")
print(f"IDs     : {IDS_OUTPUT}")
print(f"Summary : {SUMMARY_OUTPUT}")
print("=" * 70)