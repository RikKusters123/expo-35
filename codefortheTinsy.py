#include <Arduino.h>

// ------------------ CONFIG ------------------
#define RS485_DE_PIN 10
#define RS485_SERIAL Serial2
#define PI_SERIAL Serial

#define NUM_MOTORS 8
#define MOTOR_MIN_POS 0
#define MOTOR_MAX_POS 2000

// Limit switches pins
const uint8_t limitSwitchPins[NUM_MOTORS] = {2,3,4,5,6,7,8,9};

// ------------------ MOTOR VARIABLES ------------------
long motorTargets[NUM_MOTORS];      // commanded target positions
long motorCurrentPos[NUM_MOTORS];   // current motor positions
long velocity_T = 20;               // translation max steps per batch
long velocity_R = 10;               // rotation max steps per batch

// ------------------ SETUP ------------------
void setup() {
  PI_SERIAL.begin(115200);
  RS485_SERIAL.begin(115200);
  pinMode(RS485_DE_PIN, OUTPUT);
  digitalWrite(RS485_DE_PIN, LOW);

  for (int i=0; i<NUM_MOTORS; i++) {
    pinMode(limitSwitchPins[i], INPUT_PULLUP);
    motorTargets[i] = 0;
    motorCurrentPos[i] = 0;
  }

  PI_SERIAL.println("Teensy Foosball Controller Ready with Velocity Control");
}

// ------------------ HELPER FUNCTIONS ------------------

// Send batch command to RS485 (motorCurrentPos[])
void sendBatchMotorCommands() {
  uint8_t packet[1 + NUM_MOTORS*2 + 1]; // STX + 2 bytes per motor + ETX
  packet[0] = 0x02; // STX
  for (int i=0;i<NUM_MOTORS;i++) {
    // Limit switch safety
    bool switchPressed = digitalRead(limitSwitchPins[i]) == LOW;
    if (switchPressed && motorCurrentPos[i] < MOTOR_MIN_POS) {
      motorCurrentPos[i] = MOTOR_MIN_POS;
      PI_SERIAL.print("Motor "); PI_SERIAL.print(i+1); PI_SERIAL.println(" blocked by limit switch");
    }

    // Clamp
    long pos = max(MOTOR_MIN_POS, min(MOTOR_MAX_POS, motorCurrentPos[i]));
    packet[1 + i*2] = (pos >> 8) & 0xFF;
    packet[1 + i*2 + 1] = pos & 0xFF;
  }
  packet[sizeof(packet)-1] = 0x03; // ETX

  // Send over RS485
  digitalWrite(RS485_DE_PIN,HIGH);
  delayMicroseconds(10);
  RS485_SERIAL.write(packet, sizeof(packet));
  RS485_SERIAL.flush();
  digitalWrite(RS485_DE_PIN,LOW);
}

// Update motor positions based on velocity
void updateMotorPositions() {
  for (int i=0;i<NUM_MOTORS;i++) {
    long target = motorTargets[i];
    long current = motorCurrentPos[i];
    long delta = target - current;
    long step = 0;

    if (i<4) { // translation
      step = max(-velocity_T, min(velocity_T, delta));
    } else {   // rotation
      step = max(-velocity_R, min(velocity_R, delta));
    }

    motorCurrentPos[i] += step;
  }
  sendBatchMotorCommands();
}

// ------------------ SERIAL PARSING ------------------
void parsePiCommand(String line) {
  line.trim();
  // Velocity command: VEL,T,R
  if (line.startsWith("VEL")) {
    int first = line.indexOf(',');
    int second = line.indexOf(',', first+1);
    if (first != -1 && second != -1) {
      velocity_T = line.substring(first+1,second).toInt();
      velocity_R = line.substring(second+1).toInt();
      PI_SERIAL.print("Set velocities: T="); PI_SERIAL.print(velocity_T);
      PI_SERIAL.print(", R="); PI_SERIAL.println(velocity_R);
    }
    return;
  }

  int first = line.indexOf(',');
  int second = line.indexOf(',', first+1);
  if (first == -1 || second == -1) return;

  uint8_t motorID = line.substring(0, first).toInt();
  char axis = line.substring(first+1, second).charAt(0);
  long position = line.substring(second+1).toInt();

  // Map bar+axis to motor index 0-7
  int idx = 0;
  if (axis=='T') idx = motorID - 1;        // translation 0-3
  if (axis=='R') idx = motorID + 3;        // rotation 4-7

  if (idx>=0 && idx<NUM_MOTORS) motorTargets[idx] = position;
}

void checkPiCommands() {
  static String inputLine = "";
  while (PI_SERIAL.available()) {
    char c = PI_SERIAL.read();
    if (c=='\n') {
      parsePiCommand(inputLine);
      inputLine = "";
    } else {
      inputLine += c;
    }
  }
}

// ------------------ MAIN LOOP ------------------
void loop() {
  checkPiCommands();
  updateMotorPositions();
}
