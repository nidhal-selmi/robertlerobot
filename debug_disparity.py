#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual disparity debug tool.

Displays raw disparity map, an ROI window, and computes smoothed 3D
coordinates from that ROI. Use this script when depth detection is not
working as expected to visualize the stereo pipeline.
"""

import cv2
import numpy as np
from collections import deque

# -----------------------------------------------------------------------------
# Stereo intrinsics/extrinsics (same values as in new_detect.py)
# -----------------------------------------------------------------------------
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

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, (w, h), R_lr, T_lr, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)

# Stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,
    preFilterCap=31,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def capture_frame(idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Camera {idx} capture failed")
    return frame


def compute_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    gL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disp16 = stereo.compute(gL, gR)
    cv2.filterSpeckles(disp16, 0, 400, 32)
    disp = disp16.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def disparity_to_colormap(disp: np.ndarray) -> np.ndarray:
    dv = disp.copy()
    dv[np.isnan(dv)] = np.nanmin(dv)
    dv = ((dv - dv.min()) / (dv.max() - dv.min()) * 255).astype(np.uint8)
    return cv2.applyColorMap(dv, cv2.COLORMAP_JET)


def roi_average_point(pts3d: np.ndarray, roi: tuple) -> np.ndarray | None:
    x, y, w_, h_ = roi
    region = pts3d[y : y + h_, x : x + w_]
    valid = np.all(np.isfinite(region), axis=2)
    if np.any(valid):
        return region[valid].mean(axis=0)
    return None


# -----------------------------------------------------------------------------
# Main debug loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Press 'r' to select ROI, 'q' to quit.")

    # Use center ROI as default
    roi_w, roi_h = 50, 50
    roi = ((w - roi_w) // 2, (h - roi_h) // 2, roi_w, roi_h)
    history = deque(maxlen=5)  # smoothing window

    while True:
        left = cv2.remap(capture_frame(2), mapLx, mapLy, cv2.INTER_CUBIC)
        right = cv2.remap(capture_frame(0), mapRx, mapRy, cv2.INTER_CUBIC)
        disp = compute_disparity(left, right)
        pts3d = cv2.reprojectImageTo3D(disp, Q)

        pt = roi_average_point(pts3d, roi)
        if pt is not None:
            history.append(pt)
        if history:
            smoothed = np.mean(history, axis=0)
        else:
            smoothed = None

        vis_left = left.copy()
        x, y, rw, rh = roi
        cv2.rectangle(vis_left, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        if smoothed is not None:
            cv2.putText(
                vis_left,
                f"X:{smoothed[0]:.3f} Y:{smoothed[1]:.3f} Z:{smoothed[2]:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        disp_color = disparity_to_colormap(disp)

        cv2.imshow("Left with ROI", vis_left)
        cv2.imshow("Right", right)
        cv2.imshow("Disparity", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            roi = cv2.selectROI("Left with ROI", left, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("ROI selector")
            history.clear()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
