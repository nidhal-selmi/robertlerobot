#include <AccelStepper.h>
#include <math.h>

//-------------------------
// Motor Speed and Acceleration Settings
//-------------------------

// X axis (leadscrew motor: pins 5,6,7)
const int motorSpeedX = 2000;        // steps/sec (higher torque)
const int motorAccelerationX = 4000;

// Y axis (stepper motor: pins 8,9,10)
const int motorSpeedY = 2000;        // steps/sec (faster)
const int motorAccelerationY = 4000;

// Z axis (remaining stepper: pins 11,12,13)
const int motorSpeedZ = 4000;
const int motorAccelerationZ = 4000;

//-------------------------
// Pin Definitions
//-------------------------

// X Axis (Motor3: leadscrew)
const int stepPinY = 6;
const int dirPinY  = 7;
const int enaPinY  = 5;  // active LOW

// Y Axis (Motor1)
const int stepPinX = 9;
const int dirPinX  = 10;
const int enaPinX  = 8;  // active LOW

// Z Axis (Motor2)
const int stepPinZ = 13;
const int dirPinZ  = 12;
const int enaPinZ  = 11; // active LOW

//-------------------------
// Create AccelStepper Objects
//-------------------------
AccelStepper stepperX(AccelStepper::DRIVER, stepPinX, dirPinX);
AccelStepper stepperY(AccelStepper::DRIVER, stepPinY, dirPinY);
AccelStepper stepperZ(AccelStepper::DRIVER, stepPinZ, dirPinZ);

//-------------------------
// Movement Flags and Timeouts
//-------------------------
bool movingXY = false;
bool movingZ  = false;
unsigned long startXY = 0;
unsigned long startZ  = 0;
const unsigned long timeoutXY = 500000;
const unsigned long timeoutZ  = 100000;

//-------------------------
// Setup
//-------------------------
void setup() {
  Serial.begin(9600);
  Serial.println("Ready.");

  // Enable pins LOW
  pinMode(enaPinX, OUTPUT); digitalWrite(enaPinX, LOW);
  pinMode(enaPinY, OUTPUT); digitalWrite(enaPinY, LOW);
  pinMode(enaPinZ, OUTPUT); digitalWrite(enaPinZ, LOW);

  // Configure speeds
  stepperX.setMaxSpeed(motorSpeedX);
  stepperX.setAcceleration(motorAccelerationX);

  stepperY.setMaxSpeed(motorSpeedY);
  stepperY.setAcceleration(motorAccelerationY);

  stepperZ.setMaxSpeed(motorSpeedZ);
  stepperZ.setAcceleration(motorAccelerationZ);
}

//-------------------------
// Main Loop
//-------------------------
void loop() {
  // Read commands when idle
  if (Serial.available() > 0 && !movingXY && !movingZ) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("MOVE ")) {
      // MOVE <deltaX> <deltaY>
      int i1 = cmd.indexOf(' ');
      int i2 = cmd.indexOf(' ', i1 + 1);
      if (i1 > 0 && i2 > 0) {
        int dx = cmd.substring(i1 + 1, i2).toInt();
        int dy = cmd.substring(i2 + 1).toInt();
        // X ? stepperX, Y ? stepperY
        stepperX.moveTo(stepperX.currentPosition() + dx);
        stepperY.moveTo(stepperY.currentPosition() + dy);
        movingXY = true;
        startXY = millis();
      }
    }
    else if (cmd.startsWith("MOVE_Z ")) {
      int sp = cmd.indexOf(' ');
      if (sp > 0) {
        int dz = cmd.substring(sp + 1).toInt();
        stepperZ.moveTo(stepperZ.currentPosition() + dz);
        movingZ = true;
        startZ = millis();
      }
    }
    else if (cmd == "STOP") {
      stepperX.stop(); stepperY.stop(); stepperZ.stop();
      Serial.println("Stopped.");
    }
  }

  // Run XY (X and Y together)
  if (movingXY) {
    stepperX.run();
    stepperY.run();
    if (stepperX.distanceToGo() == 0 && stepperY.distanceToGo() == 0) {
      movingXY = false;
      Serial.println("XY done.");
    }
    if (millis() - startXY > timeoutXY) {
      stepperX.stop(); stepperY.stop();
      Serial.println("XY timeout.");
      movingXY = false;
    }
  }

  // Run Z
  if (movingZ) {
    stepperZ.run();
    if (stepperZ.distanceToGo() == 0) {
      movingZ = false;
      Serial.println("Z done.");
    }
    if (millis() - startZ > timeoutZ) {
      stepperZ.stop();
      Serial.println("Z timeout.");
      movingZ = false;
    }
  }
}
