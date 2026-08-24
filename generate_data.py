import cv2
import os
VIDEO_PATHS = ["con.mp4","conveyor.mp4"]
OUTPUT_DIR = "dataset/raw_frames"
os.makedirs(OUTPUT_DIR,exist_ok=True)
frame_step = 15
image_number = 0
for video_path in VIDEO_PATHS:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        continue
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_step == 0:
            filename = os.path.join(
                OUTPUT_DIR,
                f"frame_{image_number:05d}.jpg"
            )
            cv2.imwrite(filename,frame)
            image_number += 1
        frame_number += 1
    cap.release()
print(f"Extracted {image_number} images.")