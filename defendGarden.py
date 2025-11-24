import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from adafruit_servokit import ServoKit
from gpiozero import LED
import time
import os
from collections import deque
from datetime import datetime

# to add: text updates, reset back to center after shooting
# -------------------------
# Servo setup
# -------------------------
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500, 2550)  # Pan
kit.servo[1].set_pulse_width_range(500, 2550)  # Tilt

pan_channel = 0
tilt_channel = 1
PAN_MIN, PAN_MAX = 0, 175
TILT_MIN, TILT_MAX = 80, 175

PAN_CENTER = (PAN_MIN + PAN_MAX) // 2
TILT_CENTER = 135
pan_angle = PAN_CENTER
tilt_angle = TILT_CENTER
kit.servo[pan_channel].angle = pan_angle
kit.servo[tilt_channel].angle = tilt_angle

# Servo movement parameters
TOLERANCE = 25
MAX_JUMP = 8
GAIN = 0.04

# -------------------------
# Camera setup
# -------------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (384, 384)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

frame_height, frame_width = 384, 384
center_x = frame_width // 2
center_y = frame_height // 2 - 65

# -------------------------
# MOSFET setup
# -------------------------
bang = LED(17)
last_fire = 0
FIRE_COOLDOWN = 2  # seconds

# -------------------------
# YOLO model with optimized settings
# -------------------------
model = YOLO("yolo11n_ncnn_model")

# -------------------------
# Detection parameters (IMPROVED)
# -------------------------
TARGET_CLASSES = {0, 15, 16, 17}  # person, bird, cat, dog
CLASS_NAMES = {0: "person", 15: "bird", 16: "cat", 17: "dog"}

MIN_CONFIDENCE = 0.30  # Detection threshold
MIN_BOX_AREA = 1500    # Minimum pixels to filter tiny detections
MAX_BOX_AREA = 120000  # Maximum pixels to filter unrealistic large detections

# Temporal filtering: require consistent detections
DETECTION_BUFFER_SIZE = 5  # Track last 5 frames
MIN_DETECTIONS_TO_TRACK = 2  # At least 2 out of 5 frames (was 3)
detection_buffer = deque(maxlen=DETECTION_BUFFER_SIZE)

# No stability check - fire immediately when centered
centered_frame_count = 0

# -------------------------
# Detection folders and frame capture
# -------------------------
base_folder = "detections"
os.makedirs(base_folder, exist_ok=True)

# Frame buffer to save post-fire frames
FRAMES_TO_SAVE_AFTER_FIRE = 10
post_fire_buffer = deque(maxlen=FRAMES_TO_SAVE_AFTER_FIRE)
frames_to_save_count = 0
current_detection_folder = None

# -------------------------
# Helper function: Validate detection quality
# -------------------------
def is_valid_detection(x1, y1, x2, y2, conf, class_id):
    """Apply validation checks to reduce false positives"""
    
    # Check if target class
    if class_id not in TARGET_CLASSES:
        return False, "Not target class"
    
    # Check confidence
    if conf < MIN_CONFIDENCE:
        return False, "Low confidence"
    
    # Calculate dimensions
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Check area bounds
    if area < MIN_BOX_AREA:
        return False, f"Too small (area={area:.0f})"
    if area > MAX_BOX_AREA:
        return False, f"Too large (area={area:.0f})"
    
    return True, "Valid"

# -------------------------
# Main loop
# -------------------------
try:
    print("Starting animal tracker - fires immediately when centered...")
    print(f"Target classes: {[CLASS_NAMES[c] for c in TARGET_CLASSES]}")
    print(f"Min confidence: {MIN_CONFIDENCE}, Tolerance: {TOLERANCE}px")
    print(f"Fire cooldown: {FIRE_COOLDOWN}s")
    
    while True:
        frame = picam2.capture_array()
        frame_display = frame.copy()
        
        # Add frame to post-fire buffer (always keep last 10 frames ready)
        post_fire_buffer.append(frame_display.copy())
        
        # If we're in post-fire saving mode, save frames
        if frames_to_save_count > 0:
            frame_filename = os.path.join(current_detection_folder, f"frame_{FRAMES_TO_SAVE_AFTER_FIRE - frames_to_save_count + 1:02d}.jpg")
            cv2.imwrite(frame_filename, frame_display)
            frames_to_save_count -= 1
            if frames_to_save_count == 0:
                print(f"Saved {FRAMES_TO_SAVE_AFTER_FIRE} frames to {current_detection_folder}")
                current_detection_folder = None

        # YOLO inference with more aggressive settings
        results = model.predict(
            frame, 
            imgsz=384, 
            verbose=False,
            conf=MIN_CONFIDENCE,  # Pre-filter at model level
            iou=0.5,  # Non-max suppression threshold
            agnostic_nms=False
        )
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Filter for target animals and apply validation
        valid_detections = []
        for i in range(len(boxes)):
            class_id = int(classes[i])
            if class_id in TARGET_CLASSES:
                x1, y1, x2, y2 = boxes[i]
                is_valid, reason = is_valid_detection(x1, y1, x2, y2, confs[i], class_id)
                
                if is_valid:
                    valid_detections.append((boxes[i], confs[i], class_id))
                else:
                    # Draw rejected detections in red for debugging
                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                    cv2.putText(frame_display, f"X: {reason}", (int(x1), int(y1)-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    # Print debug info
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    print(f"REJECTED: {CLASS_NAMES[class_id]} - {reason} | Conf: {confs[i]:.2f}, Area: {area:.0f}, Box: ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")
        
        # Track immediately if any valid detection exists
        if len(valid_detections) > 0:
            # Track largest valid target
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box, conf, class_id in valid_detections]
            i = areas.index(max(areas))
            (x1, y1, x2, y2), conf, class_id = valid_detections[i]
            
            obj_x = int((x1 + x2) / 2)
            obj_y = int((y1 + y2) / 2)

            # Draw valid detection in green
            cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_display, f"{CLASS_NAMES[class_id]} {conf:.2f}", (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center crosshair
            size = 6
            cv2.line(frame_display, (obj_x - size, obj_y), (obj_x + size, obj_y), (0, 0, 255), 2)
            cv2.line(frame_display, (obj_x, obj_y - size), (obj_x, obj_y + size), (0, 0, 255), 2)

            # Compute error
            error_x = obj_x - center_x
            error_y = obj_y - center_y

            # PAN
            if abs(error_x) > TOLERANCE:
                delta = np.clip(error_x * GAIN, -MAX_JUMP, MAX_JUMP)
                pan_angle -= delta
                pan_angle = max(PAN_MIN, min(PAN_MAX, pan_angle))
                kit.servo[pan_channel].angle = pan_angle

            # TILT
            if abs(error_y) > TOLERANCE:
                delta = np.clip(error_y * GAIN, -MAX_JUMP, MAX_JUMP)
                tilt_angle += delta
                tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))
                kit.servo[tilt_channel].angle = tilt_angle

            # Check if centered and fire immediately
            current_time = time.time()
            if abs(error_x) < TOLERANCE and abs(error_y) < TOLERANCE:
                centered_frame_count += 1
                
                # Display centered status
                cv2.putText(frame_display, f"CENTERED - READY TO FIRE", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Fire immediately if cooldown period has passed
                if current_time - last_fire > FIRE_COOLDOWN and frames_to_save_count == 0:
                    # Create detection folder with class_date_time naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_name = f"{CLASS_NAMES[class_id]}_{timestamp}"
                    current_detection_folder = os.path.join(base_folder, folder_name)
                    os.makedirs(current_detection_folder, exist_ok=True)
                    
                    # Save all buffered frames immediately
                    for idx, buffered_frame in enumerate(post_fire_buffer):
                        frame_filename = os.path.join(current_detection_folder, f"frame_{idx+1:02d}.jpg")
                        cv2.imwrite(frame_filename, buffered_frame)
                    
                    # Set counter to save next frames
                    frames_to_save_count = FRAMES_TO_SAVE_AFTER_FIRE
                    
                    # Fire the gun
                    bang.blink(on_time=1, off_time=0, n=1)
                    last_fire = current_time
                    print(f"FIRING at {CLASS_NAMES[class_id]}! Conf: {conf:.2f}, Area: {areas[i]:.0f} | Folder: {folder_name}")
            else:
                centered_frame_count = 0
        else:
            # No detection - reset counters
            centered_frame_count = 0

        # Draw target center
        cv2.circle(frame_display, (center_x, center_y), 5, (255, 0, 0), 2)
        cv2.circle(frame_display, (center_x, center_y), TOLERANCE, (255, 0, 0), 1)

        # Display FPS and stats
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0
        cv2.putText(frame_display, f'FPS: {fps:.1f}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Person Tracking", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cv2.destroyAllWindows()
    bang.off()
    kit.servo[pan_channel].angle = PAN_CENTER
    kit.servo[tilt_channel].angle = TILT_CENTER
