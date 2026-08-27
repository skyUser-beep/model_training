import cv2
import numpy as np

# CONFIGURATION
VIDEO_PATH = "finetune.mp4"
# Detection resolution
# 0.35 means detection happens at about 35% of original size
DETECTION_SCALE = 0.35
# Re-learn the color clusters every N frames
RELEARN_EVERY = 20
# Number of color clusters
K = 4
# Small regions are ignored
MIN_AREA = 80
# Status stability
PRESENT_FRAMES_REQUIRED = 3
EMPTY_FRAMES_REQUIRED = 10
# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
print("Video opened successfully.")
present_counter = 0
empty_counter = 0
current_status = "INITIALIZING..."
frame_number = 0
# Cluster centers learned from previous frame
cluster_centers = None
# LEARN COLOR CLUSTERS
def learn_clusters(frame):
    # Small image for fast processing
    small = cv2.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv2.INTER_AREA
    )
    # LAB works better than BGR for color separation
    lab = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2LAB
    )
    pixels = lab.reshape(
        (-1, 3)
    ).astype(np.float32)
    # K-means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        15,
        1.0
    )
    # Sample pixels instead of using every pixel
    # This makes K-means much faster.
    max_samples = 15000
    if len(pixels) > max_samples:
        indices = np.random.choice(
            len(pixels),
            max_samples,
            replace=False
        )
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    _, _, centers = cv2.kmeans(
        pixels_sample,
        K,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)
# CREATE MASK USING EXISTING CLUSTERS
def create_mask(frame, centers):
    # Work at low resolution
    small = cv2.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv2.INTER_AREA
    )
    lab = cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)
    # Find closest cluster
    # Instead of constructing a huge NxK distance matrix,
    # calculate each cluster separately.
    # This is much more memory efficient.
    best_distance = np.full(len(pixels),np.inf,dtype=np.float32)
    best_cluster = np.zeros(len(pixels),dtype=np.uint8)
    for i, center in enumerate(centers):
        diff = pixels - center
        distance = np.sum(diff * diff,axis=1)
        update = distance < best_distance
        best_distance[update] = distance[update]
        best_cluster[update] = i
    labels = best_cluster.reshape(
        lab.shape[:2])
    # Automatically choose biscuit cluster
    # Convert centers to BGR
    center_img = centers.reshape(1,K,3).astype(np.uint8)
    center_bgr = cv2.cvtColor(center_img,cv2.COLOR_LAB2BGR)[0]
    scores = []
    for center in center_bgr:
        b = int(center[0])
        g = int(center[1])
        r = int(center[2])
        brightness = (r + g + b) / 3
        warmth = (r + g - 2 * b)
        score = (brightness * 0.5 +warmth * 0.5)
        scores.append(score)
    biscuit_cluster = int(
        np.argmax(scores)
    )
    # Create binary mask
    mask = np.zeros(
        labels.shape,
        dtype=np.uint8
    )
    mask[
        labels == biscuit_cluster
    ] = 255
    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )
    return mask
# PROCESS VIDEO
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break
    frame_number += 1
    # Keep original display size
    frame = cv2.resize(
        frame,
        (848, 480),
        interpolation=cv2.INTER_AREA
    )
    # Learn clusters only occasionally
    if (cluster_centers is None or frame_number % RELEARN_EVERY == 0):
        cluster_centers = learn_clusters(frame)
        # Create mask
    mask = create_mask(
        frame,
        cluster_centers
    )
    # Resize mask for display
    display_mask = cv2.resize(mask,(frame.shape[1], frame.shape[0]),interpolation=cv2.INTER_NEAREST)
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        display_mask,
        connectivity=8
    )
    clean_mask = np.zeros_like(
        display_mask
    )
    for i in range(1, num_labels):
        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]
        if area >= MIN_AREA:
            clean_mask[
                labels == i
            ] = 255
            display_mask = clean_mask
    # Calculate coverage
    biscuit_pixels = cv2.countNonZero(
        display_mask
    )
    total_pixels = (
        display_mask.shape[0]
        * display_mask.shape[1]
    )
    coverage = (
        biscuit_pixels
        / total_pixels
    )
    # Find contours
    contours, _ = cv2.findContours(
        display_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    object_count = 0
    for contour in contours:
        area = cv2.contourArea(
            contour
        )
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(
            contour
        )
        if w < 10 or h < 10:
            continue
        object_count += 1
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Biscuit {object_count}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )
    # STATUS
    if object_count > 0:
        present_counter += 1
        empty_counter = 0
        if present_counter >= PRESENT_FRAMES_REQUIRED:
            current_status = ("BISCUITS PRESENT")
    else:
        empty_counter += 1
        present_counter = 0
        if empty_counter >= EMPTY_FRAMES_REQUIRED:
            current_status = (
                "CONVEYOR EMPTY"
            )
    # STATUS COLOR
    if current_status == "BISCUITS PRESENT":
        status_color = (0, 255, 0)
    elif current_status == "CONVEYOR EMPTY":
        status_color = (0, 0, 255)
    else:
        status_color = (0, 255, 255)
    # STATUS DISPLAY
    cv2.rectangle(
        frame,
        (15, 15),
        (650, 95),
        (30, 30, 30),
        -1
    )
    cv2.putText(
        frame,
        current_status,
        (40, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        status_color,
        3
    )
    # DEBUG INFORMATION
    cv2.putText(
        frame,
        f"Coverage: {coverage * 100:.1f}%",
        (15, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (15, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    # SHOW
    cv2.imshow(
        "Biscuit Detection",
        frame
    )
    cv2.imshow(
        "Automatic Biscuit Mask",
        display_mask
    )
    # QUIT
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")