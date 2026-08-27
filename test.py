import os
import time
import cv2
from urllib.parse import quote
from ultralytics import YOLO

# 0. FORCE FFMPEG TO USE TCP (Fixes 30s stream timeout)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# HIKVISION + YOLO V2 LIVE TEST
# 1. CAMERA SETTINGS
CAMERA_IP = "192.168.160.20"
USERNAME = "admin"
PASSWORD = "CCTVM_P@ssw0rd@2"  # If password contains @ or %, quote handles it safely below

ENCODED_USERNAME = quote(USERNAME, safe="")
ENCODED_PASSWORD = quote(PASSWORD, safe="")

# Hikvision main stream
RTSP_URL = (
    f"rtsp://{ENCODED_USERNAME}:{ENCODED_PASSWORD}@"
    f"{CAMERA_IP}:554/Streaming/Channels/102"
)

# 2. YOLO MODEL
MODEL_PATH = (
    r"C:\Users\Sahil\Downloads\WelcomeScreen"
    r"\runs\detect\runs\detect\biscuit_v2"
    r"\weights\best.pt"
)

# 3. DETECTION SETTINGS
CONFIDENCE_THRESHOLD = 0.60
PRESENT_FRAMES_REQUIRED = 3
EMPTY_FRAMES_REQUIRED = 10
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2

# 4. LOAD MODEL
print("=" * 60)
print("HIKVISION + YOLO V2 LIVE TEST")
print("=" * 60)
print("\nLoading model:")
print(MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded successfully.\n")


# 5. CAMERA CONNECTION FUNCTION
def connect_camera():
    print("Connecting to Hikvision...")
    safe_url = RTSP_URL.replace(ENCODED_PASSWORD, "********")
    print(safe_url)

    # Explicitly force CAP_FFMPEG backend to avoid CAP_IMAGES fallback
    camera = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not camera.isOpened():
        print("Could not open Hikvision RTSP stream.")
        camera.release()
        return None
    print("Hikvision camera connected successfully!")
    return camera


# 6. OPEN CAMERA
cap = connect_camera()
if cap is None:
    print("\nERROR: Initial camera connection failed.")
    print("\nCheck:")
    print("1. Camera IP pingability")
    print("2. RTSP enabled in Hikvision web UI (Configuration -> Network -> Advanced -> Integration Protocol)")
    print("3. Password credentials")
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
    ret, frame = cap.read()

    if not ret or frame is None:
        print("\nWARNING: Frame not received from Hikvision.")
        cap.release()
        reconnect_attempts += 1
        if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
            print("ERROR: Maximum reconnect attempts reached.")
            break
        print(f"Reconnecting... attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}")
        time.sleep(RECONNECT_DELAY)
        cap = connect_camera()
        if cap is None:
            continue
        print("Stream reconnected successfully.")
        continue

    reconnect_attempts = 0
    frame_number += 1

    # YOLO DETECTION
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    biscuit_count = 0
    confidences = []

    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence >= CONFIDENCE_THRESHOLD:
                biscuit_count += 1
                confidences.append(confidence)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Biscuit {confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
                )

    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

    # STABLE DECISION LOGIC
    if biscuit_count > 0:
        present_counter += 1
        empty_counter = 0
        if present_counter >= PRESENT_FRAMES_REQUIRED:
            current_status = "BISCUITS PRESENT"
    else:
        empty_counter += 1
        present_counter = 0
        if empty_counter >= EMPTY_FRAMES_REQUIRED:
            current_status = "CONVEYOR EMPTY"

    # FPS CALCULATION
    current_time = time.perf_counter()
    elapsed = current_time - last_time
    last_time = current_time
    current_fps = (1.0 / elapsed) if elapsed > 0 else 0

    fps_history.append(current_fps)
    if len(fps_history) > 20:
        fps_history.pop(0)
    display_fps = sum(fps_history) / len(fps_history)

    # UI PANEL
    status_color = (0, 255, 0) if current_status == "BISCUITS PRESENT" else (0, 0,
                                                                             255) if current_status == "CONVEYOR EMPTY" else (
        0, 255, 255)

    cv2.rectangle(frame, (10, 10), (500, 150), (25, 25, 25), -1)
    cv2.putText(frame, current_status, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Biscuits: {biscuit_count}", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f"Avg confidence: {average_confidence:.2f}", (25, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {display_fps:.1f}", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imshow("Hikvision - Biscuit Detection", frame)

    if frame_number % 30 == 0:
        print(
            f"Frame {frame_number} | Biscuits: {biscuit_count} | Avg conf: {average_confidence:.2f} | Status: {current_status}")

    # Corrected key mask 0xFF
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Q pressed. Stopping...")
        break

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 60 + "\nCAMERA TEST FINISHED\n" + "=" * 60)