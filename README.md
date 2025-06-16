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
are displayed in separate windows.  After the initial stereo-based movement, the
script switches to a camera mounted on the gripper (index&nbsp;4) to centre the
strawberry before advancing the gripper.

The repository also includes ``detect_apriltag_left.py`` and
``calibrate_tilt_apriltag.py`` for working with AprilTags. These helpers are
configured for the ``tagStandard36h11`` family and expect a tag measuring
36.5&nbsp;mm across.

Additional Arduino sketches would control the mechanical actuators that move the
robotic arm and gripper.


