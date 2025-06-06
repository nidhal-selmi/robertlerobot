#!/usr/bin/env python3
"""Calibrate camera tilt using an AprilTag marker.

This script detects an AprilTag using the on-board cameras and
computes the rotation angles (roll, pitch, yaw) that transform
coordinates from the camera frame to the gripper frame.

It requires OpenCV with the `aruco` module and access to a camera.
"""

import cv2
import numpy as np
import math
import time

# -----------------------------------------------------------------------------
# Camera intrinsics (same values as in ``new_detect.py``)
# -----------------------------------------------------------------------------
# Left camera
fxL, fyL, cxL, cyL = 746.065, 736.904, 342.529, 294.211
K_L = np.array([[fxL, 0, cxL], [0, fyL, cyL], [0, 0, 1]])
D_L = np.array([-0.459493, 0.118473, -0.0158825, -0.00770142, 0.198851])
# Right camera
fxR, fyR, cxR, cyR = 746.248, 726.204, 322.339, 314.147
K_R = np.array([[fxR, 0, cxR], [0, fyR, cyR], [0, 0, 1]])
D_R = np.array([-0.379353, -0.359228, -0.0293038, -0.0031526, 1.04])

# -----------------------------------------------------------------------------
# AprilTag detector configuration
# -----------------------------------------------------------------------------
TAG_SIZE = 0.05  # Tag side length in meters (adjust to your printed tag)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to XYZ Euler angles in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.rad2deg([x, y, z])


def capture_frame(idx: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Grab a single frame from the specified camera."""
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Camera {idx} capture failed")
    return frame


def estimate_tilt(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Detect an AprilTag in ``frame`` and return roll/pitch/yaw angles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None:
        raise RuntimeError("No AprilTag detected")

    # Use the first detected tag
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, TAG_SIZE, K, D)
    rvec = rvecs[0].reshape(3)
    R, _ = cv2.Rodrigues(rvec)

    # Assuming the tag frame matches the gripper frame
    angles = rotation_matrix_to_euler(R)
    return angles


if __name__ == "__main__":
    print("Capturing frames for tilt calibration...")
    left_frame = capture_frame(2)
    right_frame = capture_frame(0)

    try:
        angles_left = estimate_tilt(left_frame, K_L, D_L)
        angles_right = estimate_tilt(right_frame, K_R, D_R)
    except RuntimeError as e:
        print(e)
    else:
        print("Left camera -> gripper angles (deg): ", angles_left)
        print("Right camera -> gripper angles (deg):", angles_right)

