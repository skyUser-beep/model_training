# from ultralytics import YOLO
# import cv2
# import csv
# import numpy as np
# # Your NEW fine-tuned model
# MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_fine_tuned\weights\best.pt"
# # Put your NEW/UNSEEN test video here
# VIDEO_PATH = r"v1.mp4"
# # Output video
# OUTPUT_VIDEO = r"test_results.mp4"
# # CSV report
# CSV_PATH = r"test_results.csv"
# # Detection confidence threshold
# CONFIDENCE = 0.50
# # Image size used by YOLO
# IMAGE_SIZE = 640
# # LOAD MODEL
# print()
# print("=" * 60)
# print("BISCUIT MODEL TEST")
# print("=" * 60)
# print()
# print("Loading model:")
# print(MODEL_PATH)
# model = YOLO(MODEL_PATH)
# print("Model loaded successfully.")
# print()
#
# # OPEN VIDEO
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print("ERROR: Could not open video:")
#     print(VIDEO_PATH)
#     raise SystemExit
# # Video information
# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# if fps <= 0:
#     fps = 25
# print("Video opened successfully.")
# print(f"Resolution : {width} x {height}")
# print(f"FPS        : {fps:.2f}")
# print(f"Frames     : {total_frames}")
# print()
# # OUTPUT VIDEO
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(
#     OUTPUT_VIDEO,
#     fourcc,
#     fps,
#     (width, height)
# )
# # CSV FILE
# csv_file = open(
#     CSV_PATH,
#     "w",
#     newline=""
# )
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow([
#     "frame",
#     "time_seconds",
#     "biscuits_detected",
#     "average_confidence",
#     "minimum_confidence",
#     "maximum_confidence",
#     "low_confidence_count"
# ])
# # STATISTICS
# frame_number = 0
# all_counts = []
# all_confidences = []
# max_count = 0
# min_count = None
#
# # PROCESS VIDEO
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_number += 1
#     # YOLO DETECTION
#     results = model.predict(
#         source=frame,
#         conf=CONFIDENCE,
#         imgsz=IMAGE_SIZE,
#         device="cpu",
#         verbose=False
#     )
#     result = results[0]
#     # GET CONFIDENCES
#     confidences = []
#     if result.boxes is not None:
#         for conf in result.boxes.conf:
#             confidences.append(float(conf))
#     # COUNT BISCUITS
#     biscuit_count = len(confidences)
#     # CONFIDENCE STATISTICS
#     if len(confidences) > 0:
#         average_confidence = float(np.mean(confidences))
#         minimum_confidence = float(np.min(confidences))
#         maximum_confidence = float(np.max(confidences))
#         low_confidence_count = sum(conf < 0.60
#             for conf in confidences
#         )
#     else:
#         average_confidence = 0.0
#         minimum_confidence = 0.0
#         maximum_confidence = 0.0
#         low_confidence_count = 0
#     # SAVE STATISTICS
#     all_counts.append(biscuit_count)
#     all_confidences.extend(confidences)
#     if biscuit_count > max_count:
#         max_count = biscuit_count
#     if min_count is None or biscuit_count < min_count:
#         min_count = biscuit_count
#     time_seconds = frame_number / fps
#     csv_writer.writerow([
#         frame_number,
#         f"{time_seconds:.2f}",
#         biscuit_count,
#         f"{average_confidence:.3f}",
#         f"{minimum_confidence:.3f}",
#         f"{maximum_confidence:.3f}",
#         low_confidence_count
#     ])
#     # DRAW YOLO RESULTS
#     annotated = result.plot()
#     # ADD INFORMATION PANEL
#     cv2.rectangle(
#         annotated,
#         (0, 0),
#         (430, 145),
#         (0, 0, 0),
#         -1
#     )
#     cv2.putText(
#         annotated,
#         f"Biscuits: {biscuit_count}",
#         (15, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 0),
#         2
#     )
#     cv2.putText(
#         annotated,
#         f"Avg confidence: {average_confidence:.2f}",
#         (15, 60),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (255, 255, 255),
#         2
#     )
#     cv2.putText(
#         annotated,
#         f"Min confidence: {minimum_confidence:.2f}",
#         (15, 88),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (255, 255, 255),
#         2
#     )
#     cv2.putText(
#         annotated,
#         f"Max confidence: {maximum_confidence:.2f}",
#         (15, 116),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (255, 255, 255),
#         2
#     )
#     cv2.putText(
#         annotated,
#         f"Low conf (<0.60): {low_confidence_count}",
#         (15, 142),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.55,
#         (0, 200, 255),
#         2
#     )
#     # SHOW VIDEO
#     cv2.imshow(
#         "Biscuit Model Test",
#         annotated
#     )
#     # SAVE VIDEO
#     out.write(annotated)
#     # PRINT EVERY 30 FRAMES
#     if frame_number % 30 == 0:
#         print(f"Frame {frame_number}/{total_frames} | "
#             f"Biscuits: {biscuit_count} | "
#             f"Avg conf: {average_confidence:.2f} | "
#             f"Low conf: {low_confidence_count}")
#     # Press Q to stop
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         print()
#         print("Stopped by user.")
#         break
# # CLEANUP
# cap.release()
# out.release()
# csv_file.close()
# cv2.destroyAllWindows()
#
# # FINAL STATISTICS
# print()
# print("=" * 60)
# print("TEST FINISHED")
# print("=" * 60)
# print()
# processed_frames = len(all_counts)
# if processed_frames > 0:
#     average_count = float(np.mean(all_counts))
#     median_count = float(np.median(all_counts))
#     count_std = float(np.std(all_counts))
# else:
#     average_count = 0
#     median_count = 0
#     count_std = 0
# if len(all_confidences) > 0:
#     overall_average_confidence = float(np.mean(all_confidences))
# else:
#     overall_average_confidence = 0
# print(f"Frames processed       : {processed_frames}")
# print(f"Average biscuit count  : {average_count:.2f}")
# print(f"Median biscuit count   : {median_count:.2f}")
# print(f"Minimum biscuit count  : {min_count}")
# print(f"Maximum biscuit count  : {max_count}")
# print(f"Count standard dev.    : {count_std:.2f}")
# print(f"Overall average conf.  : "f"{overall_average_confidence:.3f}")
# print()
# print("Output video:")
# print(OUTPUT_VIDEO)
# print()
# print("CSV report:")
# print(CSV_PATH)
# print()
# print("=" * 60)
# print("TEST COMPLETE")
# print("=" * 60)

from ultralytics import YOLO, trackers
import cv2
MODEL_PATH= r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_v3\weights\best.pt"
VIDEO_PATH="v4.mp4"

CONFIDENCE=0.30
LINE_Y=100

model=YOLO(MODEL_PATH)
print("Model Loaded Successfully")
cap=cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video")
    exit()
print("Video Loaded Successfully")

total_biscuits=0
count_ids=set()
prev_pos={}

while True:
    ret,frame =cap.read()
    if not ret:
        break

    results=model.track(frame,conf=CONFIDENCE
                        , persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False)
    result=results[0]
    annotated_frame=result.plot()
    if result.boxes is not None and result.boxes.id is not None:
        boxes=result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box , track_id in zip(boxes,track_ids):
            x1,y1,x2,y2=box

            center_x=int((x1+x2)/2)
            center_y=int((y1+y2)/2)

            prev_y=prev_pos.get(track_id)

            if prev_y is not None:

                if prev_y > LINE_Y >= center_y:
                    if track_id not in count_ids:
                        total_biscuits+=1
                        count_ids.add(track_id)
                        print(f"Biscuit counted ! ID:{track_id} total: {total_biscuits}")

            prev_pos[track_id]=center_y
            cv2.putText(annotated_frame,
                        f"ID: {track_id}",
                        (int(x1),int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2)

    cv2.line(annotated_frame,
             (0,LINE_Y),
             (frame.shape[1],LINE_Y),
             (0,0,255),
             3)
    cv2.putText(annotated_frame,
                "Counting Line",
                (20,LINE_Y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2)
    cv2.putText(annotated_frame,
                f"Total Biscuits: {total_biscuits}",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,255,0),
                3)
    cv2.imshow("Biscuits Counting",
               annotated_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
           break

cap.release()
cv2.destroyAllWindows()
print("*",*50)
print(f"TOTAL BISCUITS : {total_biscuits}")
print("*",*50)