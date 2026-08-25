import cv2 #openCV- opening video, reading frames, saving frames as images
import os # create output directory, construct image filenames safely
VIDEO_PATHS = ["con.mp4","conveyor.mp4"] # path
OUTPUT_DIR = "dataset/raw_frames"  # save things
os.makedirs(OUTPUT_DIR,exist_ok=True) #create folder
frame_step = 15 # save upto 15 frames
image_number = 0 # image numbering for saving
for video_path in VIDEO_PATHS:
    cap = cv2.VideoCapture(video_path) #open the video
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        continue
    frame_number = 0
    while True:
        ret, frame = cap.read() # ret- frame was successfully read, frame- will contains actual image
        if not ret: # cannot read any frame by OpenCV
            break
        if frame_number % frame_step == 0:
            filename = os.path.join(
                OUTPUT_DIR,
                f"frame_{image_number:05d}.jpg"
            ) # create complete path
            cv2.imwrite(filename,frame) # frame to disk as JPEG image
            image_number += 1
        frame_number += 1
    cap.release() # finished
print(f"Extracted {image_number} images.")

