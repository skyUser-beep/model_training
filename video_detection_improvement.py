import cv2
import numpy as np
import time
from collections import deque


# CONFIGURATION
VIDEO_PATH = "conveyor.mp4"
# Processing is done at this scale.
# Display remains at the original video resolution.
DETECTION_SCALE = 0.35
# How many frames are used for temporal smoothing
HISTORY_LENGTH = 12
# Stable status
PRESENT_FRAMES_REQUIRED = 3
EMPTY_FRAMES_REQUIRED = 10
# Recalculate appearance model occasionally
MODEL_UPDATE_INTERVAL = 30
# OPEN VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30.0
print("Video opened successfully.")
print(f"Video FPS: {video_fps:.2f}")

# STATE
frame_number = 0
present_counter = 0
empty_counter = 0
current_status = "INITIALIZING..."
coverage_history = deque(
    maxlen=HISTORY_LENGTH
)
object_history = deque(
    maxlen=HISTORY_LENGTH
)
# Automatically learned biscuit shape
learned_width = None
learned_height = None
learned_aspect = None
# FPS
fps_history = deque(maxlen=20)
last_time = time.perf_counter()

# UTILITY
def robust_percentile(values, percentile, default):
    """
    Safe percentile calculation.
    """
    values = np.asarray(values)
    values = values[
        np.isfinite(values)
    ]
    if len(values) == 0:
        return default
    return float(
        np.percentile(
            values,
            percentile
        )
    )
# CREATE ADAPTIVE BISCUIT MASK
def create_biscuit_mask(frame):
    # Resize for fast processing
    small = cv2.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv2.INTER_AREA
    )
    # LAB COLOR SPACE
    lab = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2LAB
    )
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)
    # CHROMA / WARMTH
    # We don't specify "biscuit = HSV X".
    # Instead we calculate the warm/cool difference
    # from the image itself.
    warmth = B - A
    warmth = cv2.GaussianBlur(
        warmth,
        (7, 7),
        0
    )
    # NORMALIZE WARMTH
    warmth_norm = cv2.normalize(
        warmth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    # Automatic threshold
    warmth_threshold, _ = cv2.threshold(
        warmth_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Prevent extreme threshold values
    low_limit = np.percentile(
        warmth_norm,
        45
    )
    high_limit = np.percentile(
        warmth_norm,
        75
    )
    warmth_threshold = np.clip(
        warmth_threshold,
        low_limit,
        high_limit
    )
    color_mask = (
        warmth_norm >= warmth_threshold
    ).astype(np.uint8) * 255
    # LOCAL TEXTURE
    # Biscuits have embossed texture.
    # We calculate local standard deviation automatically.
    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    local_mean = cv2.blur(
        gray,
        (11, 11)
    )
    local_mean_sq = cv2.blur(
        gray * gray,
        (11, 11)
    )
    local_std = np.sqrt(
        np.maximum(
            local_mean_sq -
            local_mean * local_mean,
            0
        )
    )
    texture_norm = cv2.normalize(
        local_std,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    texture_threshold = cv2.threshold(
        texture_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[0]
    texture_mask = (
        texture_norm >= texture_threshold
    ).astype(np.uint8) * 255
    # COMBINE INFORMATION
    # Color is primary.
    # Texture is used to suppress some conveyor regions.
    combined = cv2.bitwise_and(color_mask,texture_mask)
    # If AND gives too little information,
    # fall back to color information.
    color_pixels = cv2.countNonZero(color_mask)

    combined_pixels = cv2.countNonZero(combined)
    if (color_pixels > 0 and combined_pixels < color_pixels * 0.10):
        combined = color_mask
    # MORPHOLOGICAL CLEANUP
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=2
    )
    # UPSCALE MASK
    mask = cv2.resize(
        combined,
        (
            frame.shape[1],
            frame.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )
    return mask
# CLEAN COMPONENTS
def clean_components(mask):
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )
    clean = np.zeros_like(mask)

    image_area = (
        mask.shape[0] *
        mask.shape[1]
    )
    # Automatically derive a minimum area
    # from the image size rather than a biscuit-specific
    # pixel count.
    minimum_area = image_area * 0.00015
    # Don't allow absurdly tiny values
    minimum_area = max(
        minimum_area,
        30
    )
    for i in range(1,num_labels):
        area = stats[i,cv2.CC_STAT_AREA]
        if area >= minimum_area:
            clean[labels == i] = 255
    return clean
# FIND CANDIDATES

def find_candidates(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    image_area = (mask.shape[0] *mask.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle_area = w * h
        if rectangle_area <= 0:
            continue
        # Shape measurements
        aspect = (max(w, h) /max(min(w, h), 1))
        rectangularity = (area /rectangle_area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = (area /hull_area)
        else:
            solidity = 0
        # Reject obviously useless regions
        # These limits are intentionally broad.
        # They are not biscuit dimensions.
        if w < 8 or h < 8:
            continue
        if area < image_area * 0.00008:
            continue
        # Huge region covering most of image
        if area > image_area * 0.35:
            continue
        # Extremely thin lines
        if aspect > 12:
            continue
        # Very irregular regions
        if solidity < 0.35:
            continue
        candidates.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "aspect": aspect,
            "rectangularity": rectangularity,
            "solidity": solidity
        })
    return candidates
# LEARN TYPICAL BISCUIT SHAPE
def learn_shape(candidates):
    global learned_width
    global learned_height
    global learned_aspect
    if len(candidates) < 3:
        return
    widths = np.array([
        c["w"]
        for c in candidates
    ], dtype=np.float32)
    heights = np.array([
        c["h"]
        for c in candidates
    ], dtype=np.float32)
    aspects = np.array([
        c["aspect"]
        for c in candidates
    ], dtype=np.float32)
    # Robust statistics
    median_width = np.median(widths)
    median_height = np.median(heights)
    median_aspect = np.median(aspects)
    # Slowly update instead of suddenly changing
    if learned_width is None:
        learned_width = median_width
        learned_height = median_height
        learned_aspect = median_aspect
    else:
        alpha = 0.10
        learned_width = ((1 - alpha) * learned_width+ alpha * median_width)
        learned_height = ((1 - alpha) * learned_height+ alpha * median_height)
        learned_aspect = ((1 - alpha) * learned_aspect+ alpha * median_aspect)

# FILTER CANDIDATES USING LEARNED SHAPE
def filter_candidates(candidates):
    if learned_width is None:
        return candidates
    good = []
    for c in candidates:
        # Compare candidate to learned dimensions.
        # Wide tolerance is intentional because perspective
        # causes biscuits to change apparent size.
        width_ratio = (c["w"] / max(learned_width, 1))
        height_ratio = (c["h"] /max(learned_height, 1))
        aspect_difference = abs(c["aspect"] -learned_aspect)
        # Accept a broad range.
        if (0.25 <= width_ratio <= 4.0 and 0.25 <= height_ratio <= 4.0 and aspect_difference <= max(
                learned_aspect * 0.8,
                1.0
            )
        ):
            good.append(c)
    return good
# CALCULATE OCCUPANCY
def calculate_occupancy(mask):
    total = mask.size
    if total == 0:
        return 0.0
    white = cv2.countNonZero(mask)
    return white / total
# UPDATE STATUS
def update_status(occupancy,candidate_count):
    global present_counter
    global empty_counter
    global current_status
    coverage_history.append(occupancy)
    object_history.append(candidate_count)
    # Temporal average
    average_coverage = np.mean(coverage_history)
    average_objects = np.mean(object_history)
    # Adaptive decision
    # We use both:
    # 1. detected area
    # 2. detected candidates
    # This prevents a single false contour from declaring
    # the whole conveyor occupied.
    present = (average_objects >= 1 and average_coverage > 0.01)
    if present:
        present_counter += 1
        empty_counter = 0
        if (present_counter >=  PRESENT_FRAMES_REQUIRED):
            current_status = ("BISCUITS PRESENT")
    else:
        empty_counter += 1
        present_counter = 0
        if (empty_counter >=EMPTY_FRAMES_REQUIRED):
            current_status = ("CONVEYOR EMPTY")
    return (average_coverage,average_objects)

# DRAW
def draw_detection(frame,candidates,average_coverage, average_objects,fps):
    # Status color
    if current_status == "BISCUITS PRESENT":
        status_color = (0,255,0)
    elif current_status == "CONVEYOR EMPTY":
        status_color = (0,0,255)
    else:
        status_color = (0,255,255)
    # Draw candidate boxes
    for number, c in enumerate(candidates,start=1):
        x = c["x"]
        y = c["y"]
        w = c["w"]
        h = c["h"]
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Biscuit {number}",
            (
                x,
                max(
                    y - 5,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    # Status background
    cv2.rectangle(
        frame,
        (15, 15),
        (650, 100),
        (25, 25, 25),
        -1
    )
    cv2.putText(
        frame,
        current_status,
        (40, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        status_color,
        3,
        cv2.LINE_AA
    )
    # Information
    cv2.putText(
        frame,
        f"Coverage: {average_coverage * 100:.1f}%",
        (15, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Objects: {int(round(average_objects))}",
        (15, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    if learned_width is not None:
        cv2.putText(
            frame,
            (f"Learned shape: "f"{learned_width:.0f}x"f"{learned_height:.0f}"),
            (15, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break
    frame_number += 1
    # Keep original video resolution for display
    original_h, original_w = (frame.shape[:2])
    # DETECTION
    mask = create_biscuit_mask(frame)
    # Clean mask
    clean_mask = clean_components(mask)
    # Candidate detection
    candidates = find_candidates(clean_mask)
    # Learn shape periodically
    if (frame_number == 1 or frame_number % MODEL_UPDATE_INTERVAL == 0):
        learn_shape(candidates)
    # Apply learned shape
    filtered_candidates = (filter_candidates(candidates))
    # Occupancy
    coverage = calculate_occupancy(clean_mask)
    # Status
    (average_coverage,average_objects) = update_status(coverage,len(filtered_candidates))
    # FPS
    current_time = time.perf_counter()
    elapsed = (current_time -last_time)
    last_time = current_time
    if elapsed > 0:
        current_fps = (1.0 /elapsed)
    else:
        current_fps = 0
    fps_history.append(current_fps)
    display_fps = np.mean(fps_history)

    # DRAW
    draw_detection(
        frame,
        filtered_candidates,
        average_coverage,
        average_objects,
        display_fps
    )
    # SHOW
    cv2.imshow("Biscuit Detection", frame)
    cv2.imshow("Adaptive Biscuit Mask",clean_mask)
    # QUIT
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# CLEANUP
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")