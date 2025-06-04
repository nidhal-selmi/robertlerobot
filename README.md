# Strawberry Harvesting Robot

This repository contains code for a simple strawberry harvesting robot. The system
combines an Arduino microcontroller and a Raspberry Pi running Python.

The Arduino controls the hardware components such as servos and sensors that
physically pick the strawberries. The Raspberry Pi runs Python scripts to detect
ripe strawberries and coordinate the harvesting sequence.

`detection.py` is intended for the Raspberry Pi and will handle image capture
and detection logic.

Additional Arduino sketches would control the mechanical actuators that move the
robotic arm and gripper.


