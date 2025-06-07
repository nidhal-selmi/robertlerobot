# Strawberry Harvesting Robot

This repository contains code for a simple strawberry harvesting robot. The system
combines an Arduino microcontroller and a Raspberry Pi running Python.

The Arduino controls the hardware components such as servos and sensors that
physically pick the strawberries. The Raspberry Pi runs Python scripts to detect
ripe strawberries and coordinate the harvesting sequence.

`new_detect.py` runs on the Raspberry Pi and shows a local interface using
OpenCV. Press **n** to capture a burst of stereo frames and run detection. The
script averages the 3D positions of the detected markers over several frames for
more stable measurements. The left camera, right camera and detection overlay
are displayed in separate windows.

Additional Arduino sketches would control the mechanical actuators that move the
robotic arm and gripper.


