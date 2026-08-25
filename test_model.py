from ultralytics import YOLO
import cv2
# CONFIGURATION
MODEL_PATH = r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\runs\detect\biscuit_fine_tuned\weights\best.pt"
VIDEO_PATH = r"finetune.mp4"
CONFIDENCE = 0.30

# LOAD MODEL
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
print("Video opened successfully.")
# PROCESS VIDEO
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # YOLO detection
    results = model.predict(frame,
        conf=CONFIDENCE,
        verbose=False
    )
    # Draw detections
    annotated_frame = results[0].plot()
    # Number of detected biscuits
    boxes = results[0].boxes
    if boxes is not None:
        biscuit_count = len(boxes)
    else:
        biscuit_count = 0
    # Display count
    cv2.putText(annotated_frame,
        f"Biscuits: {biscuit_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("YOLO Biscuit Detection",
        annotated_frame
    )
    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
print("Detection finished.")