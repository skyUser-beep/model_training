import cv2
import time
from urllib.parse import quote
from ultralytics import YOLO

# hikvision + YOLO V2 LIVE TEST
# 1. CAMERA SETTINGS
CAMERA_IP = "192.168.160.20"
USERNAME = "admin"
PASSWORD = "CCTVM_P@ssw0rd@2"
ENCODED_USERNAME = quote(USERNAME, safe="")
ENCODED_PASSWORD = quote(PASSWORD, safe="")
# hikvision main stream
RTSP_URL = (
    f"rtsp://{ENCODED_USERNAME}:{ENCODED_PASSWORD}@"
    f"{CAMERA_IP}:554/Streaming/Channels/102"
)
# 2. YOLO MODEL
MODEL_PATH = (
   r"C:\Users\Sahil\Downloads\WelcomeScreen\runs\detect\biscuit_v3\weights\best.pt"
)
# 3. DETECTION SETTINGS
# Start with 0.60 for the empty-table test.
# You can later compare 0.40 vs 0.50 vs 0.60.
CONFIDENCE_THRESHOLD = 0.50
PRESENT_FRAMES_REQUIRED = 3
EMPTY_FRAMES_REQUIRED = 10
# How many times we retry the camera
MAX_RECONNECT_ATTEMPTS = 5
# Delay between reconnect attempts
RECONNECT_DELAY = 2
# 4. LOAD MODEL
print("=" * 60)
print("hikvision + YOLO V2 LIVE TEST")
print("=" * 60)
print()
print("Loading model:")
print(MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")
print()

# 5. CAMERA CONNECTION FUNCTION
def connect_camera():
    print("Connecting to hikvision...")
    # Hide password in console
    safe_url = RTSP_URL.replace(
        ENCODED_PASSWORD,
        "********"
    )
    print(safe_url)
    camera = cv2.VideoCapture(RTSP_URL)
    if not camera.isOpened():
        print("Could not open hikvision RTSP stream.")
        camera.release()
        return None
    print("hikvision camera connected successfully!")
    return camera
# 6. OPEN CAMERA
cap = connect_camera()
if cap is None:
    print()
    print("ERROR: Initial camera connection failed.")
    print()
    print("Check:")
    print("1. Camera IP")
    print("2. Username")
    print("3. Password")
    print("4. Ethernet connection")
    print("5. RTSP enabled")
    print("6. Port 554")
    print("7. hikvision network settings")
    exit()
# 7. STATE
present_counter = 0
empty_counter = 0
current_status = "INITIALIZING..."
frame_number = 0
fps_history = []
last_time = time.perf_counter()
reconnect_attempts = 0
# 8. MAIN LOOP
while True:
    # READ FRAME
    ret, frame = cap.read()
    # HANDLE FAILED FRAME
    if not ret or frame is None:
        print()
        print("WARNING: Frame not received from hikvision.")
        cap.release()
        reconnect_attempts += 1
        if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
            print("ERROR: Maximum reconnect attempts reached.")
            break
        print(f"Reconnecting... "
            f"attempt {reconnect_attempts}/"
            f"{MAX_RECONNECT_ATTEMPTS}"
        )
        time.sleep(RECONNECT_DELAY)
        cap = connect_camera()
        if cap is None:
            continue
        print("Stream reconnected successfully.")
        continue
    # Successful frame
    reconnect_attempts = 0
    frame_number += 1
    # YOLO DETECTION
    results = model.predict(frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )
    result = results[0]
    # COUNT DETECTIONS
    biscuit_count = 0
    confidences = []
    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence >= CONFIDENCE_THRESHOLD:
                biscuit_count += 1
                confidences.append(confidence)
                # Bounding box
                x1, y1, x2, y2 = map(int,box.xyxy[0])
                # Draw box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                # Confidence label
                label = (
                    f"Biscuit "
                    f"{confidence:.2f}"
                )
                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 5, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
    # AVERAGE CONFIDENCE
    if confidences:
        average_confidence = (sum(confidences)/ len(confidences))
    else:
        average_confidence = 0.0
    # STABLE PRESENT / EMPTY DECISION
    if biscuit_count > 0:
        present_counter += 1
        empty_counter = 0
        if present_counter>= PRESENT_FRAMES_REQUIRED:
            current_status = "BISCUITS PRESENT"
    else:
        empty_counter += 1
        present_counter = 0
        if empty_counter>= EMPTY_FRAMES_REQUIRED:
            current_status = "CONVEYOR EMPTY"
    # FPS
    current_time = time.perf_counter()
    elapsed = (current_time- last_time)
    last_time = current_time
    if elapsed > 0:
        current_fps = (1.0 / elapsed)
    else:
        current_fps = 0
    fps_history.append(current_fps)
    if len(fps_history) > 20:
        fps_history.pop(0)
    display_fps = (sum(fps_history)/ len(fps_history))
    # STATUS COLOR
    if current_status == "BISCUITS PRESENT":
        status_color = (0,255,0 )
    elif current_status == "CONVEYOR EMPTY":
        status_color = (0,0,255)
    else:
        status_color = (0,255,255)
    # INFORMATION PANEL
    cv2.rectangle(frame,(10, 10),(500, 150),(25, 25, 25),-1)
    cv2.putText(frame,current_status,(25, 50),cv2.FONT_HERSHEY_SIMPLEX,0.85,status_color, 2,cv2.LINE_AA)
    cv2.putText(frame,f"Biscuits: {biscuit_count}",(25, 82),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2,cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Avg confidence: " f"{average_confidence:.2f}",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"FPS: {display_fps:.1f}",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    # SHOW VIDEO
    cv2.imshow(
        "hikvision - Biscuit Detection",
        frame
    )
    # PRINT EVERY 30 FRAMES
    if frame_number % 30 == 0:
        print(
            f"Frame {frame_number} | "
            f"Biscuits: {biscuit_count} | "
            f"Avg conf: "
            f"{average_confidence:.2f} | "
            f"Status: {current_status}"
        )
    # QUIT
    key = cv2.waitKey(1) & 0xF
    if key == ord("q"):
        print("Q pressed. Stopping...")
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
print()
print("=" * 60)
print("CAMERA TEST FINISHED")
print("=" * 60)