#!/usr/bin/env python3
"""Preview the left camera feed and highlight AprilTag detections.

This script opens the left camera, attempts to detect an AprilTag and displays
the result live. It is configured for the ``tagStandard36h11`` family and
expects a printed tag that is 36.5 mm wide. Press 'q' in the display window to
quit.
"""

import cv2
import numpy as np
import time

CAM_INDEX = 2  # device index for the left camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TAG_SIZE_M = 0.0365  # tag side length in meters (36.5 mm)

# Intrinsic parameters for the left camera
fxL, fyL, cxL, cyL = 709.46, 708.781, 361.165, 274.045
K_L = np.array([[fxL, 0, cxL], [0, fyL, cyL], [0, 0, 1]])
D_L = np.array([-0.297109, -1.06046, -0.00744828, -0.0171288, 3.58896])

# AprilTag detector using OpenCV's aruco module (tagStandard36h11 family).
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    time.sleep(0.2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = DETECTOR.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, TAG_SIZE_M, K_L, D_L
            )
            for rvec, tvec, c in zip(rvecs, tvecs, corners):
                cv2.aruco.drawDetectedMarkers(frame, [c])
                cv2.drawFrameAxes(frame, K_L, D_L, rvec, tvec, TAG_SIZE_M * 0.5)
        else:
            cv2.putText(
                frame,
                "No tag detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Left Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
