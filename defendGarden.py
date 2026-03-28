import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from adafruit_servokit import ServoKit
from gpiozero import LED
import time
import os
import threading
from collections import deque
from datetime import datetime


def run_loop(state: dict, frame_lock: threading.Lock) -> None:
    """
    Main garden defender loop. Runs in a background thread.

    state keys (read/write):
        running (bool)       – set False to stop
        mode (str)           – "auto" | "manual"
        latest_frame (bytes) – latest JPEG, written each frame
        moving_pan (int)     – -1 / 0 / 1, continuous pan direction (manual)
        moving_tilt (int)    – -1 / 0 / 1, continuous tilt direction (manual)
        manual_fire (bool)   – True to fire once (manual)
    """

    # ── Servo setup ───────────────────────────────────────────────────────
    kit = ServoKit(channels=16)
    kit.servo[0].set_pulse_width_range(500, 2550)  # Pan
    kit.servo[1].set_pulse_width_range(500, 2550)  # Tilt

    pan_channel = 0
    tilt_channel = 1
    PAN_MIN,  PAN_MAX  = 0, 175
    TILT_MIN, TILT_MAX = 80, 175
    PAN_CENTER  = (PAN_MIN + PAN_MAX) // 2
    TILT_CENTER = 135
    pan_angle  = PAN_CENTER
    tilt_angle = TILT_CENTER
    kit.servo[pan_channel].angle  = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle

    TOLERANCE = 25
    MAX_JUMP  = 8
    GAIN      = 0.04
    MANUAL_SPEED = 3  # degrees per frame in manual mode

    # ── Camera setup ──────────────────────────────────────────────────────
    picam2 = Picamera2()
    picam2.preview_configuration.main.size   = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    YOLO_SIZE = 384
    center_x  = YOLO_SIZE // 2
    center_y  = YOLO_SIZE // 2 - 65

    # ── MOSFET / gun setup ────────────────────────────────────────────────
    bang      = LED(17)
    last_fire = 0
    FIRE_COOLDOWN = 2  # seconds

    # ── YOLO model ────────────────────────────────────────────────────────
    model_path = os.environ.get("GARDEFENDER_MODEL", "yolo11n_ncnn_model")
    model = YOLO(model_path)

    # ── Detection parameters ──────────────────────────────────────────────
    CLASS_NAMES = {
        0: "person", 15: "bird", 16: "cat", 17: "dog",
        18: "horse", 19: "sheep", 20: "cow", 21: "elephant",
        22: "bear", 23: "zebra", 24: "giraffe",
    }
    MIN_CONFIDENCE = 0.65
    MIN_BOX_AREA   = 500
    MAX_BOX_AREA   = 200000

    centered_frame_count = 0

    # ── Motion gate ───────────────────────────────────────────────────────
    bg_subtractor  = cv2.createBackgroundSubtractorMOG2(
        history=50, varThreshold=40, detectShadows=False
    )
    MOTION_MIN_AREA = 500
    IDLE_SLEEP_S    = 0.15

    # ── Detection saving ──────────────────────────────────────────────────
    base_folder = "detections"
    os.makedirs(base_folder, exist_ok=True)
    FRAMES_TO_SAVE       = 10
    post_fire_buffer     = deque(maxlen=FRAMES_TO_SAVE)
    frames_to_save_count = 0
    current_detection_folder = None

    def is_valid_detection(x1, y1, x2, y2, conf, class_id, target_classes):
        if class_id not in target_classes:
            return False, "Not target class"
        if conf < MIN_CONFIDENCE:
            return False, "Low confidence"
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            return False, f"Too small (area={area:.0f})"
        if area > MAX_BOX_AREA:
            return False, f"Too large (area={area:.0f})"
        return True, "Valid"

    def encode_frame(frame_bgr):
        """Encode a BGR frame as JPEG bytes."""
        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()

    print("Garden defender loop starting…")

    _last_frame_time = time.time()

    try:
        while state.get("running", True):
            _now = time.time()
            _dt  = _now - _last_frame_time
            if _dt > 0:
                state["fps"] = 1.0 / _dt
            _last_frame_time = _now

            frame       = picam2.capture_array()
            frame_display = frame.copy()
            frame_small = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            scale_x     = frame.shape[1] / YOLO_SIZE
            scale_y     = frame.shape[0] / YOLO_SIZE

            post_fire_buffer.append(frame_display.copy())

            # Write servo angles back to state for stats display
            state["pan_angle"]  = pan_angle
            state["tilt_angle"] = tilt_angle

            # Save post-fire frames
            if frames_to_save_count > 0:
                idx = FRAMES_TO_SAVE - frames_to_save_count + 1
                cv2.imwrite(
                    os.path.join(current_detection_folder, f"frame_{idx:02d}.jpg"),
                    frame_display,
                )
                frames_to_save_count -= 1
                if frames_to_save_count == 0:
                    print(f"Saved {FRAMES_TO_SAVE} frames to {current_detection_folder}")
                    current_detection_folder = None

            # ── MANUAL MODE ───────────────────────────────────────────────
            if state.get("mode") == "manual":
                pan_dir  = state.get("moving_pan",  0)
                tilt_dir = state.get("moving_tilt", 0)

                if pan_dir:
                    pan_angle = max(PAN_MIN,  min(PAN_MAX,  pan_angle  + pan_dir  * MANUAL_SPEED))
                    kit.servo[pan_channel].angle = pan_angle

                if tilt_dir:
                    tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle + tilt_dir * MANUAL_SPEED))
                    kit.servo[tilt_channel].angle = tilt_angle

                if state.get("manual_fire"):
                    current_time = time.time()
                    if current_time - last_fire > FIRE_COOLDOWN:
                        bang.blink(on_time=1, off_time=0, n=1)
                        last_fire = current_time
                        print("Manual fire!")
                    state["manual_fire"] = False

                cv2.putText(frame_display, "MANUAL", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                with frame_lock:
                    state["latest_frame"] = encode_frame(frame_display)
                time.sleep(1 / 30)
                continue

            # ── AUTO MODE — motion gate ────────────────────────────────────
            fg_mask       = bg_subtractor.apply(frame_small)
            motion_pixels = cv2.countNonZero(fg_mask)
            if motion_pixels < MOTION_MIN_AREA and frames_to_save_count == 0:
                cv2.putText(frame_display, "IDLE", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                with frame_lock:
                    state["latest_frame"] = encode_frame(frame_display)
                time.sleep(IDLE_SLEEP_S)
                continue

            # ── AUTO MODE — YOLO inference ────────────────────────────────
            results = model.predict(
                frame_small,
                imgsz=YOLO_SIZE,
                verbose=False,
                conf=MIN_CONFIDENCE,
                iou=0.5,
                agnostic_nms=False,
            )

            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            target_classes = state.get("target_classes", set())
            valid_detections = []
            for i in range(len(boxes)):
                class_id = int(classes[i])
                if class_id in target_classes:
                    x1, y1, x2, y2 = boxes[i]
                    is_valid, reason = is_valid_detection(x1, y1, x2, y2, confs[i], class_id, target_classes)
                    if is_valid:
                        valid_detections.append((boxes[i], confs[i], class_id))
                    else:
                        dx1, dy1 = int(x1*scale_x), int(y1*scale_y)
                        dx2, dy2 = int(x2*scale_x), int(y2*scale_y)
                        cv2.rectangle(frame_display, (dx1, dy1), (dx2, dy2), (0, 0, 255), 1)
                        cv2.putText(frame_display, f"X: {reason}", (dx1, dy1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        area = (x2-x1)*(y2-y1)
                        print(f"REJECTED: {CLASS_NAMES[class_id]} - {reason} | "
                              f"Conf: {confs[i]:.2f}, Area: {area:.0f}")

            if valid_detections:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b, _, _ in valid_detections]
                i = areas.index(max(areas))
                (x1, y1, x2, y2), conf, class_id = valid_detections[i]

                obj_x = int((x1 + x2) / 2)
                obj_y = int((y1 + y2) / 2)

                dx1, dy1 = int(x1*scale_x), int(y1*scale_y)
                dx2, dy2 = int(x2*scale_x), int(y2*scale_y)
                cv2.rectangle(frame_display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.putText(frame_display, f"{CLASS_NAMES[class_id]} {conf:.2f}",
                            (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                dobj_x, dobj_y = int(obj_x*scale_x), int(obj_y*scale_y)
                cv2.line(frame_display, (dobj_x-10, dobj_y), (dobj_x+10, dobj_y), (0, 0, 255), 2)
                cv2.line(frame_display, (dobj_x, dobj_y-10), (dobj_x, dobj_y+10), (0, 0, 255), 2)

                error_x = obj_x - center_x
                error_y = obj_y - center_y

                if abs(error_x) > TOLERANCE:
                    pan_angle -= np.clip(error_x * GAIN, -MAX_JUMP, MAX_JUMP)
                    pan_angle = max(PAN_MIN, min(PAN_MAX, pan_angle))
                    kit.servo[pan_channel].angle = pan_angle

                if abs(error_y) > TOLERANCE:
                    tilt_angle += np.clip(error_y * GAIN, -MAX_JUMP, MAX_JUMP)
                    tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))
                    kit.servo[tilt_channel].angle = tilt_angle

                current_time = time.time()
                if abs(error_x) < TOLERANCE and abs(error_y) < TOLERANCE:
                    centered_frame_count += 1
                    cv2.putText(frame_display, "CENTERED - READY TO FIRE",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if current_time - last_fire > FIRE_COOLDOWN and frames_to_save_count == 0:
                        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                        folder_name = f"{CLASS_NAMES[class_id]}_{timestamp}"
                        current_detection_folder = os.path.join(base_folder, folder_name)
                        os.makedirs(current_detection_folder, exist_ok=True)

                        for idx, buf_frame in enumerate(post_fire_buffer):
                            cv2.imwrite(
                                os.path.join(current_detection_folder, f"frame_{idx+1:02d}.jpg"),
                                buf_frame,
                            )

                        frames_to_save_count = FRAMES_TO_SAVE
                        bang.blink(on_time=1, off_time=0, n=1)
                        last_fire = current_time
                        print(f"FIRING at {CLASS_NAMES[class_id]}! "
                              f"Conf: {conf:.2f}, Area: {areas[i]:.0f} | Folder: {folder_name}")
                else:
                    centered_frame_count = 0
            else:
                centered_frame_count = 0

            # Draw aim target
            dcenter_x = int(center_x * scale_x)
            dcenter_y = int(center_y * scale_y)
            cv2.circle(frame_display, (dcenter_x, dcenter_y), 5, (255, 0, 0), 2)
            cv2.circle(frame_display, (dcenter_x, dcenter_y), int(TOLERANCE * scale_x), (255, 0, 0), 1)

            inference_time = results[0].speed["inference"]
            fps = 1000 / inference_time if inference_time > 0 else 0
            cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with frame_lock:
                state["latest_frame"] = encode_frame(frame_display)

    except Exception as e:
        print(f"Garden defender loop error: {e}")
        raise
    finally:
        bang.off()
        kit.servo[pan_channel].angle  = PAN_CENTER
        kit.servo[tilt_channel].angle = TILT_CENTER
        picam2.stop()
        print("Garden defender loop stopped.")
