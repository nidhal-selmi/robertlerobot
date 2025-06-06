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
# 1) Interface Setup
# ------------------------------------------------
latest_frames = {'left': None, 'right': None, 'disp': None, 'vis': None}

# ------------------------------------------------
# 2) Robot & Serial Config
# ------------------------------------------------
ARDUINO_PORT = '/dev/ttyACM0'
BAUDRATE = 9600
arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)

def send_command(cmd):
    print(f">>> Sending: {cmd}")
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
TILT_X_DEG = -75.0   # upward tilt around X-axis
TILT_Y_DEG = 0.0    # placeholder for Y-axis tilt
TILT_Z_DEG = 0.0    # placeholder for Z-axis tilt
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
fxL, fyL, cxL, cyL = 746.065, 736.904, 342.529, 294.211
K_L = np.array([[fxL, 0, cxL], [0, fyL, cyL], [0, 0, 1]])
D_L = np.array([-0.459493, 0.118473, -0.0158825, -0.00770142, 0.198851])
fxR, fyR, cxR, cyR = 746.248, 726.204, 322.339, 314.147
K_R = np.array([[fxR, 0, cxR], [0, fyR, cyR], [0, 0, 1]])
D_R = np.array([-0.379353, -0.359228, -0.0293038, -0.0031526, 1.04])
R_lr = np.array([[0.99789, -0.00879122, 0.0643256], [0.0128662, 0.997917, -0.0632119], [-0.0636359, 0.0639062, 0.995925]])
T_lr = np.array([-48.7956, 0.437362, 2.72847]) / 1000.0
h, w = 480, 640

# ------------------------------------------------
# 5) Stereo rectify & matcher
# ------------------------------------------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R_lr, T_lr,
                                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5,
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
def run_cycle():
    left = cv2.remap(capture_frame(2), mapLx, mapLy, cv2.INTER_CUBIC)
    right = cv2.remap(capture_frame(0), mapRx, mapRy, cv2.INTER_CUBIC)
    gL, gR = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    d16 = stereo.compute(gL, gR)
    cv2.filterSpeckles(d16, 0, 400, 32)
    disp = d16.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    print(f"Disp stats: min={np.nanmin(disp):.2f}, max={np.nanmax(disp):.2f}, mean={np.nanmean(disp):.2f}")
    latest_frames['left'], latest_frames['right'] = left, right
    dv = disp.copy(); dv[np.isnan(dv)] = np.nanmin(dv)
    dv = ((dv - dv.min())/(dv.max()-dv.min())*255).astype(np.uint8)
    latest_frames['disp'] = cv2.applyColorMap(dv, cv2.COLORMAP_JET)
    vis = left.copy(); hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    mask_t = cv2.morphologyEx(cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
        cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    ), cv2.MORPH_OPEN, KERNEL)
    cnts = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(f"Red contours found: {len(cnts)}")
    P_t = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,ww,hh = cv2.boundingRect(c)
        cv2.rectangle(vis, (x,y), (x+ww,y+hh), (0,0,255), 2)
        P_t = pts3d[y+hh//2, x+ww//2]
        print(f"P_t coords: {P_t}")
    mask_b = cv2.morphologyEx(cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE), cv2.MORPH_OPEN, KERNEL)
    cnts_b = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(f"Blue contours found: {len(cnts_b)}")
    P_b = None
    if cnts_b:
        cb = max(cnts_b, key=cv2.contourArea)
        bx,by,bw,bbh = cv2.boundingRect(cb)
        cv2.rectangle(vis, (bx,by), (bx+bw,by+bbh), (255,0,0), 2)
        P_b = pts3d[by+bbh//2, bx+bw//2]
        print(f"P_b coords: {P_b}")
    latest_frames['vis'] = vis
    if P_t is None or P_b is None:
        print("Detection missing object(s). Next cycle.")
        return
    if not np.all(np.isfinite(P_t)) or not np.all(np.isfinite(P_b)):
        print(f"Non-finite values: P_t={P_t}, P_b={P_b}")
        return
    delta_vec = P_t - P_b
    print(f"Delta_vec before tilt: {delta_vec}")
    marker_vec = R_tilt.dot(delta_vec)
    mm = marker_vec * 1000.0
    print(f"Move (mm) in marker frame: {mm}")
    sx = int(mm[0] * STEPS_PER_MM_X)
    sy = int(mm[1] * STEPS_PER_MM_YZ)
    sz = int(mm[2] * STEPS_PER_MM_YZ)
    print(f"Steps -> X:{sx}, Y:{sy}, Z:{sz}")
    send_command(f"MOVE {sx:+d} {sy:+d}")
    send_command(f"MOVE_Z {sz:+d}")
    time.sleep(5)
    send_command(f"MOVE_Z {-sz:+d}")
    send_command(f"MOVE {-sx:+d} {-sy:+d}")


    time.sleep(0.1)

# ------------------------------------------------
# 8) Display helper
# ------------------------------------------------
def show_frames():
    if latest_frames['left'] is not None:
        cv2.imshow('Left Camera', latest_frames['left'])
    if latest_frames['right'] is not None:
        cv2.imshow('Right Camera', latest_frames['right'])
    if latest_frames['vis'] is not None:
        cv2.imshow('Detection', latest_frames['vis'])
    cv2.waitKey(1)

# ------------------------------------------------
# 9) Interactive loop
# ------------------------------------------------
if __name__ == '__main__':
    print("Press 'n' + Enter for next detection, 'q' + Enter to quit.")
    while True:
        cmd = input("Enter command (n/q): ").strip().lower()
        if cmd == 'n':
            run_cycle()
            show_frames()
        elif cmd == 'q':
            break
    arduino.close()
    cv2.destroyAllWindows()
