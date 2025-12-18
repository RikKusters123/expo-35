#include <Arduino.h>

// ------------------ CONFIG ------------------

// RS485 DE/RE pin
#define RS485_DE_PIN 10

// USB Serial to Pi
#define PI_SERIAL Serial

// RS485 Serial to motors
#define RS485_SERIAL Serial2

#define NUM_BARS 4
#define NUM_MOTORS 8

// Limit switches pins (one per motor)
const uint8_t limitSwitchPins[NUM_MOTORS] = {2,3,4,5,6,7,8,9};

// Motor targets
long motorTargets[NUM_MOTORS];

// Software min/max positions
#define MOTOR_MIN_POS 0
#define MOTOR_MAX_POS 2000

// ------------------ HELPER FUNCTIONS ------------------

// Map bar + axis to motor ID (1-8)
uint8_t getMotorID(uint8_t barID, char axis) {
  if (barID < 1 || barID > NUM_BARS) return 0;
  if (axis == 'T') return barID;       // translation motors 1-4
  if (axis == 'R') return barID + 4;   // rotation motors 5-8
  return 0;
}

// ------------------ MOTOR COMMANDS ------------------

// Send all motor positions in one batch over RS485
void sendBatchMotorCommands() {
  uint8_t packet[1 + NUM_MOTORS*2 + 1]; // STX + 2 bytes per motor + ETX
  packet[0] = 0x02; // STX

  for (int i = 0; i < NUM_MOTORS; i++) {
    // Read limit switch
    bool switchPressed = digitalRead(limitSwitchPins[i]) == LOW; // NO switches
    if (switchPressed && motorTargets[i] < MOTOR_MIN_POS) {
      motorTargets[i] = MOTOR_MIN_POS; // block movement below min
      PI_SERIAL.print("Motor "); PI_SERIAL.print(i+1); PI_SERIAL.println(" blocked by limit switch");
    }

    // Clamp position
    long pos = max(MOTOR_MIN_POS, min(MOTOR_MAX_POS, motorTargets[i]));

    // Store in packet
    packet[1 + i*2] = (pos >> 8) & 0xFF; // high byte
    packet[1 + i*2 + 1] = pos & 0xFF;    // low byte
  }

  packet[sizeof(packet)-1] = 0x03; // ETX

  // Send packet over RS485
  digitalWrite(RS485_DE_PIN, HIGH);
  delayMicroseconds(10);
  RS485_SERIAL.write(packet, sizeof(packet));
  RS485_SERIAL.flush();
  digitalWrite(RS485_DE_PIN, LOW);
}

// ------------------ UART PARSING ------------------

void parsePiCommand(String line) {
  line.trim();
  int first = line.indexOf(',');
  int second = line.indexOf(',', first + 1);
  if (first == -1 || second == -1) return;

  uint8_t barID = line.substring(0, first).toInt();
  char axis = line.substring(first + 1, second).charAt(0);
  long position = line.substring(second + 1).toInt();

  uint8_t motorID = getMotorID(barID, axis);
  if (motorID == 0) return;

  motorTargets[motorID - 1] = position;
}

void checkPiCommands() {
  static String inputLine = "";
  while (PI_SERIAL.available()) {
    char c = PI_SERIAL.read();
    if (c == '\n') {
      parsePiCommand(inputLine);
      inputLine = "";
      // Send all motors in one batch immediately after receiving command
      sendBatchMotorCommands();
    } else {
      inputLine += c;
    }
  }
}

// ------------------ SETUP ------------------

void setup() {
  PI_SERIAL.begin(115200);

  RS485_SERIAL.begin(115200); // match stepper driver baud
  pinMode(RS485_DE_PIN, OUTPUT);
  digitalWrite(RS485_DE_PIN, LOW); // start in receive mode

  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(limitSwitchPins[i], INPUT_PULLUP);
    motorTargets[i] = 0;
  }

  PI_SERIAL.println("Teensy Foosball Controller Ready (Batch Mode)");
}

// ------------------ MAIN LOOP ------------------

void loop() {
  checkPiCommands();

  // Optional: resend last batch periodically
  // static unsigned long lastSend = 0;
  // if (millis() - lastSend > 50) {
  //     sendBatchMotorCommands();
  //     lastSend = millis();
  // }
}
