#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------
# Stereo Debug Interface for Depth & Detection
# ------------------------------------------------

import cv2
import numpy as np
import serial
import time

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

def send_command(cmd):
    log(f">>> Sending: {cmd}")
    arduino.write((cmd + "\n").encode())
    arduino.flush()

# ------------------------------------------------
# 3) Settings
# ------------------------------------------------
STEPS_PER_MM_X = 115.38
STEPS_PER_MM_YZ = 18.75
LOWER_RED1 = np.array([0, 120, 70]); UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70]); UPPER_RED2 = np.array([180, 255, 255])
LOWER_BLUE = np.array([100, 150, 50]); UPPER_BLUE = np.array([140, 255, 255])
KERNEL = np.ones((5,5), np.uint8)

# Tilt angles around axes (degrees)
TILT_X_DEG = -68.0   # upward tilt around X-axis
TILT_Y_DEG = 4.0    # placeholder for Y-axis tilt
TILT_Z_DEG = 177.0    # placeholder for Z-axis tilt
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
fxL, fyL, cxL, cyL = 764.753, 759.377, 396.363, 243.605
K_L = np.array([[fxL, 0, cxL], [0, fyL, cyL], [0, 0, 1]])
D_L = np.array([-0.482866, 0.237679, 0.00102909, -0.0134808, -0.00693421])
fxR, fyR, cxR, cyR = 757.955, 750.441, 360.581, 262.167
K_R = np.array([[fxR, 0, cxR], [0, fyR, cyR], [0, 0, 1]])
D_R = np.array([-0.559392, 0.869527, -0.0131288, -0.00425451, -1.42753])
R_lr = np.array([
    [0.997201, -0.00359349, 0.0746845],
    [0.00904248, 0.997309, -0.0727507],
    [-0.0742221, 0.0732224, 0.99455],
])
T_lr = np.array([-49.0077, 1.12187, -2.45181]) / 1000.0
h, w = 480, 640

# ------------------------------------------------
# 5) Stereo rectify & matcher
# ------------------------------------------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R_lr, T_lr,
                                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                                P1=8*3*5*5, P2=32*3*5*5, disp12MaxDiff=1,
                                preFilterCap=31, uniquenessRatio=15,
                                speckleWindowSize=100, speckleRange=2,
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
# 7) Main cycle with verbose debug
NUM_FRAMES = 5

def _avg_bbox_point(pts3d, bbox):
    x, y, w_, h_ = bbox
    region = pts3d[y:y+h_, x:x+w_]
    valid = np.all(np.isfinite(region), axis=2)
    if np.any(valid):
        return region[valid].mean(axis=0)
    return None


def run_cycle(num_frames=NUM_FRAMES):
    P_t_list = []
    P_b_list = []
    last_vis = None
    for _ in range(num_frames):
        left = cv2.remap(capture_frame(2), mapLx, mapLy, cv2.INTER_CUBIC)
        right = cv2.remap(capture_frame(0), mapRx, mapRy, cv2.INTER_CUBIC)
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
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)

        mask_t = cv2.morphologyEx(
            cv2.bitwise_or(
                cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
                cv2.inRange(hsv, LOWER_RED2, UPPER_RED2),
            ),
            cv2.MORPH_OPEN,
            KERNEL,
        )
        cnts = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        log(f"Red contours found: {len(cnts)}")
        bbox_t = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            bbox_t = cv2.boundingRect(c)
            x, y, ww, hh = bbox_t
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
            pt = _avg_bbox_point(pts3d, bbox_t)
            if pt is not None:
                P_t_list.append(pt)
                log(f"P_t coords: {pt}")

        mask_b = cv2.morphologyEx(
            cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE), cv2.MORPH_OPEN, KERNEL
        )
        cnts_b = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        log(f"Blue contours found: {len(cnts_b)}")
        bbox_b = None
        if cnts_b:
            cb = max(cnts_b, key=cv2.contourArea)
            bbox_b = cv2.boundingRect(cb)
            bx, by, bw, bbh = bbox_b
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bbh), (255, 0, 0), 2)
            pb = _avg_bbox_point(pts3d, bbox_b)
            if pb is not None:
                P_b_list.append(pb)
                log(f"P_b coords: {pb}")

        last_vis = vis

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
    marker_vec = R_tilt.dot(delta_vec)
    mm = marker_vec * 1000.0
    log(f"Move (mm) in marker frame: {mm}")
    sx = int(mm[0] * STEPS_PER_MM_X)
    sy = int(mm[1] * STEPS_PER_MM_YZ)
    sz = int(mm[2] * STEPS_PER_MM_YZ)
    log(f"Steps -> X:{sx}, Y:{sy}, Z:{sz}")
    send_command(f"MOVE {sx:+d} {sy:+d}")
    send_command(f"MOVE_Z {sz:+d}")
    time.sleep(5)
    send_command(f"MOVE_Z {-sz:+d}")
    send_command(f"MOVE {-sx:+d} {-sy:+d}")

    time.sleep(0.1)

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
    cv2.namedWindow('Interface', cv2.WINDOW_NORMAL)
    display_interface()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            run_cycle()
        elif key == ord('q'):
            break
        display_interface()
    cv2.destroyAllWindows()
    arduino.close()
