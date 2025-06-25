#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------
# Stereo Debug Interface for Depth & Detection
# ------------------------------------------------

import cv2
import numpy as np
import serial
import time
import os
from ultralytics import YOLO

# ------------------------------------------------
# Logging utility to capture messages for on-screen display
# ------------------------------------------------
log_messages = []

def log(msg):
    print(msg)
    log_messages.append(str(msg))
    if len(log_messages) > 15:
        del log_messages[0]

# ------------------------------------------------
# 1) Frame Buffers
# ------------------------------------------------
latest_frames = {'left': None, 'right': None, 'disp': None, 'vis': None}

# ------------------------------------------------
# 2) Robot & Serial Config
# ------------------------------------------------
ARDUINO_PORT = '/dev/ttyACM0'
BAUDRATE = 9600
arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)

# YOLOv8 model for strawberry detection
MODEL = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))

def send_command(cmd):
    log(f">>> Sending: {cmd}")
    arduino.write((cmd + "\n").encode())
    arduino.flush()

# ------------------------------------------------
# 3) Settings
# ------------------------------------------------
STEPS_PER_MM_X = 115.38
STEPS_PER_MM_YZ = 18.75
# Number of millimetres to move per centering step
CENTER_MM = 5
CENTER_STEPS_X = int(CENTER_MM * STEPS_PER_MM_X)
CENTER_STEPS_Y = int(CENTER_MM * STEPS_PER_MM_YZ)

# AprilTag configuration (36h11 family, 36.5 mm marker)
TAG_SIZE_M = 0.0365
TAG_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
TAG_PARAMS = cv2.aruco.DetectorParameters()
TAG_DETECTOR = cv2.aruco.ArucoDetector(TAG_DICT, TAG_PARAMS)

# Clearance from gripper to the strawberry (meters)
# Positive values move the gripper away from the berry along
# the gripper's X, Y and Z axes.
CLEARANCE_GS_M = np.array([0.05, 0.03, 0.06])

# Offset from the AprilTag marker to the gripper origin (meters)
# These values describe where the gripper sits relative to the
# detected tag in the gripper coordinate frame.
APRILTAG_TO_GRIPPER_M = np.array([0.05, 0.05, 0.06])

# Tilt angles around axes (degrees)
TILT_X_DEG = 0  # upward tilt around X-axis
TILT_Y_DEG = -35    # placeholder for Y-axis tilt
TILT_Z_DEG = 0    # placeholder for Z-axis tilt
# Convert to radians
tx = np.deg2rad(TILT_X_DEG)
ty = np.deg2rad(TILT_Y_DEG)
tz = np.deg2rad(TILT_Z_DEG)
# Rotation matrices
R_x = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
R_y = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
R_z = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
R_tilt = R_z.dot(R_y).dot(R_x)

# ------------------------------------------------
# 4) Stereo intrinsics and extrinsics
# ------------------------------------------------
fxL, fyL, cxL, cyL = 709.46, 708.781, 361.165, 274.045
K_L = np.array([[fxL, 0, cxL], [0, fyL, cyL], [0, 0, 1]])
D_L = np.array([-0.297109, -1.06046, -0.00744828, -0.0171288, 3.58896])
fxR, fyR, cxR, cyR = 714.187, 709.169, 350.826, 268.824
K_R = np.array([[fxR, 0, cxR], [0, fyR, cyR], [0, 0, 1]])
D_R = np.array([-0.373188, -0.485962, -0.0138603, -0.0131833, 2.47106])
R_lr = np.array([
    [0.998918, -0.0028575, 0.0464225],
    [0.00384567, 0.999768, -0.0212111],
    [-0.0463511, 0.0213666, 0.998697],
])
T_lr = np.array([-52.1338, -0.397713, 0.764734]) / 1000.0
h, w = 480, 640

# ------------------------------------------------
# 5) Stereo rectify & matcher
# ------------------------------------------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R_lr, T_lr,
                                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=11,
                                P1=8*3*11**2, P2=32*3*11**2, disp12MaxDiff=1,
                                uniquenessRatio=10,
                                speckleWindowSize=150, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# ------------------------------------------------
# 6) Frame helper
def capture_frame(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Camera {idx} capture failed")
    return frame

# ------------------------------------------------
# Automatic centering using arrow commands from centrage.py
# ------------------------------------------------
def auto_center(cam_idx=4):
    """Automatically center the berry using simple byte commands."""
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(0.1)
    centrage_en_cours = True
    last_pos = (w // 2, h // 2)
    log("Auto centering... Press 'p' to pause or 'h' to go home.")
    while centrage_en_cours:
        ret, frame = cap.read()
        if not ret:
            continue
        results = MODEL(frame, verbose=False)[0]
        if len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            last_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        image_center_x = frame.shape[1] // 2
        image_center_y = frame.shape[0] // 2
        delta_x = last_pos[0] - image_center_x
        delta_y = last_pos[1] - image_center_y
        seuil = 50

        # Draw feedback
        cv2.circle(frame, last_pos, 5, (0, 255, 0), 2)
        cv2.drawMarker(frame, (image_center_x, image_center_y), (255, 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        latest_frames['vis'] = frame
        display_interface()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            log("Auto centering paused.")
            cap.release()
            return False
        if key == ord('h'):
            log("Return to home requested during centering.")
            cap.release()
            return 'home'

        if abs(delta_x) > seuil:
            if delta_x > 0:
                log("➡️ Bouger à droite pour centrer")
                send_command(f"MOVE {CENTER_STEPS_X:+d} 0")
            else:
                log("⬅️ Bouger à gauche pour centrer")
                send_command(f"MOVE {-CENTER_STEPS_X:+d} 0")
            time.sleep(0.2)
        elif abs(delta_y) > seuil:
            if delta_y > 0:
                log("⬇️ Bouger en bas pour centrer")
                send_command(f"MOVE 0 {CENTER_STEPS_Y:+d}")
            else:
                log("⬆️ Bouger en haut pour centrer")
                send_command(f"MOVE 0 {-CENTER_STEPS_Y:+d}")
            time.sleep(0.2)
        else:
            log("✅ Fraise centrée !")
            centrage_en_cours = False

    cap.release()
    return True

# ------------------------------------------------
# 7) Main cycle with verbose debug
NUM_FRAMES = 1

def _avg_bbox_point(pts3d, bbox):
    x, y, w_, h_ = bbox
    region = pts3d[y:y+h_, x:x+w_]
    valid = np.all(np.isfinite(region), axis=2)
    if np.any(valid):
        return region[valid].mean(axis=0)
    return None


def _is_valid_pt(pt: np.ndarray) -> bool:
    """Return True if point is finite and within 1 meter of the camera."""
    return pt is not None and np.all(np.isfinite(pt)) and np.linalg.norm(pt) < 1.0


def run_cycle(num_frames=NUM_FRAMES, return_to_start=True):
    P_t_list = []
    P_b_list = []
    last_vis = None
    while len(P_t_list) < num_frames:
        raw_left = capture_frame(2)
        raw_right = capture_frame(0)

        # Detect the AprilTag on the raw images before rectification
        grayL = cv2.cvtColor(raw_left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(raw_right, cv2.COLOR_BGR2GRAY)
        cornersL, idsL, _ = TAG_DETECTOR.detectMarkers(grayL)
        cornersR, idsR, _ = TAG_DETECTOR.detectMarkers(grayR)

        pb = None
        if idsL is not None:
            log("AprilTag detected in left image")
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                cornersL, TAG_SIZE_M, K_L, D_L
            )
            pb = tvecs[0].reshape(3)
        elif idsR is not None:
            log("AprilTag detected in right image")
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                cornersR, TAG_SIZE_M, K_R, D_R
            )
            pb = tvecs[0].reshape(3)
        else:
            log("No AprilTag detected.")
            continue

        # Rectify after successful tag detection
        left = cv2.remap(raw_left, mapLx, mapLy, cv2.INTER_CUBIC)
        right = cv2.remap(raw_right, mapRx, mapRy, cv2.INTER_CUBIC)
        gL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        d16 = stereo.compute(gL, gR)
        cv2.filterSpeckles(d16, 0, 400, 32)
        disp = d16.astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan
        pts3d = cv2.reprojectImageTo3D(disp, Q)
        log(
            f"Disp stats: min={np.nanmin(disp):.2f}, max={np.nanmax(disp):.2f}, mean={np.nanmean(disp):.2f}"
        )

        latest_frames['left'], latest_frames['right'] = left, right
        dv = disp.copy()
        dv[np.isnan(dv)] = np.nanmin(dv)
        dv = ((dv - dv.min()) / (dv.max() - dv.min()) * 255).astype(np.uint8)
        latest_frames['disp'] = cv2.applyColorMap(dv, cv2.COLORMAP_JET)

        vis = left.copy()
        results = MODEL(left, verbose=False)[0]
        log(f"YOLO detections: {len(results.boxes)}")
        pt = None
        bbox_t = None
        if len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            bbox_t = (x1, y1, x2 - x1, y2 - y1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            pt = _avg_bbox_point(pts3d, bbox_t)

        # pb already computed from the raw image detection earlier
        
        last_vis = vis
        if _is_valid_pt(pt) and _is_valid_pt(pb):
            P_t_list.append(pt)
            P_b_list.append(pb)
            log(f"P_t coords: {pt}")
            log(f"P_b coords: {pb}")
        else:
            log("Invalid or distant coordinate detected. Skipping frame.")
            continue

    latest_frames['vis'] = last_vis

    if len(P_t_list) == 0 or len(P_b_list) == 0:
        log("Detection missing object(s). Next cycle.")
        return

    P_t = np.mean(P_t_list, axis=0)
    P_b = np.mean(P_b_list, axis=0)

    if not np.all(np.isfinite(P_t)) or not np.all(np.isfinite(P_b)):
        log(f"Non-finite values: P_t={P_t}, P_b={P_b}")
        return

    delta_vec = P_t - P_b
    log(f"Delta_vec before tilt: {delta_vec}")

    # Transform to the gripper coordinate frame
    marker_vec = R_tilt.dot(delta_vec)

    # Account for the AprilTag to gripper offset and the
    # desired clearance from the berry
    gripper_vec = marker_vec - APRILTAG_TO_GRIPPER_M - CLEARANCE_GS_M

    mm = gripper_vec * 1000.0
    log(f"Move (mm) in marker frame: {mm}")
    sx = int(mm[0] * STEPS_PER_MM_X)
    sy = int(mm[1] * STEPS_PER_MM_YZ)
    sz = int(mm[2] * STEPS_PER_MM_YZ)
    log(f"Steps -> X:{sx}, Y:{sy}, Z:{sz}")

    # Move to the clearance position
    send_command(f"MOVE {sx:+d} {sy:+d}")
    send_command(f"MOVE_Z {sz:+d}")
    time.sleep(2)

    # Fine tune by centering with the gripper camera
    auto_center(cam_idx=4)

    # Advance 4 cm towards the berry
    advance_steps = int(40 * STEPS_PER_MM_YZ)
    send_command(f"MOVE_Z {advance_steps:+d}")
    time.sleep(2)

    reverse_cmds = [
        f"MOVE_Z {-advance_steps:+d}",
        f"MOVE_Z {-sz:+d}",
        f"MOVE {-sx:+d} {-sy:+d}",
    ]

    if return_to_start:
        for cmd in reverse_cmds:
            send_command(cmd)

    time.sleep(0.1)

    return reverse_cmds

# ------------------------------------------------
# 8) Display utility
# ------------------------------------------------

def compose_canvas():
    canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    if latest_frames['left'] is not None:
        canvas[0:h, 0:w] = latest_frames['left']
    if latest_frames['right'] is not None:
        canvas[0:h, w:w*2] = latest_frames['right']
    if latest_frames['vis'] is not None:
        canvas[h:h*2, 0:w] = latest_frames['vis']
    if latest_frames['disp'] is not None:
        canvas[h:h*2, w:w*2] = latest_frames['disp']
    return canvas

def draw_logs(img):
    y = 20
    for msg in log_messages[-10:]:
        cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

def display_interface():
    canvas = compose_canvas()
    draw_logs(canvas)
    cv2.imshow('Interface', canvas)


# ------------------------------------------------
# 9) Interactive loop
# ------------------------------------------------
if __name__ == '__main__':
    log("Press 'n' for next detection, 'q' to quit.")
    log("Press 'p' to pause centering or 'h' to return home.")
    cv2.namedWindow('Interface', cv2.WINDOW_NORMAL)
    reverse_cmds = []
    mode = 'idle'
    while True:
        # Continuously grab frames so the interface is populated even before
        # running a detection cycle.
        latest_frames['left'] = cv2.remap(capture_frame(0), mapLx, mapLy, cv2.INTER_CUBIC)
        latest_frames['right'] = cv2.remap(capture_frame(2), mapRx, mapRy, cv2.INTER_CUBIC)
        display_interface()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and mode == 'idle':
            reverse_cmds = run_cycle(return_to_start=False)
            log("Press 'h' to return to the original position.")
            mode = 'await_return'
        elif key == ord('h') and reverse_cmds:
            for cmd in reverse_cmds:
                send_command(cmd)
            log("Returned to original position.")
            mode = 'idle'
        elif key == ord('p'):
            log("Pause requested.")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()
    arduino.close()
