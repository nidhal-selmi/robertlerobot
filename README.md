# Strawberry Harvesting Robot

This repository contains code for a simple strawberry harvesting robot. The system
combines an Arduino microcontroller and a Raspberry Pi running Python.

The Arduino controls the hardware components such as servos and sensors that
physically pick the strawberries. The Raspberry Pi runs Python scripts to detect
ripe strawberries and coordinate the harvesting sequence.

`new_detect.py` runs on the Raspberry Pi and opens a single window using
OpenCV. Press **n** to capture a new stereo pair and run detection. The window
shows the left camera, right camera and detection overlay in a grid layout.

Additional Arduino sketches would control the mechanical actuators that move the
robotic arm and gripper.


