// ==============================
// SenseOn_ESP32.ino
// ==============================

#include "config.h"

#include "protocol.h"

#include "motor_control.h"

#include "display_control.h"

#include "ble_handler.h"


void setup() {

  Serial.begin(115200);

  delay(1000);


  Serial.println();
  Serial.println(
    "======================"
  );

  Serial.println(
    "    SenseOn ESP32"
  );

  Serial.println(
    "======================"
  );


  setupMotor();

  setupDisplay();

  setupBLE();


  Serial.println(
    "[SYSTEM] READY"
  );
}


void loop() {

  // BLE 연결 끊김 안전 처리
  if (
    !isPiConnected() &&
    isMotorActive()
  ) {

    stopMotors();


    Serial.println(
      "[SAFETY] BLE disconnected -> OFF"
    );
  }


  // 데이터가 1초 이상 안 오면 모터 OFF
  if (
    isMotorActive() &&
    millis() - getLastPacketTime()
      > DATA_TIMEOUT_MS
  ) {

    stopMotors();


    Serial.println(
      "[SAFETY] Data timeout -> OFF"
    );
  }


  // 새 데이터 없으면 넘어감
  if (!hasNewPacket()) {

    delay(5);

    return;
  }


  String packet =
    getLatestPacket();


  Serial.print(
    "[PACKET] "
  );

  Serial.println(
    packet
  );


  HazardData hazard =
    parsePacket(packet);


  if (!hazard.valid) {

    Serial.println(
      "[ERROR] Invalid packet"
    );

    return;
  }


  Serial.println(
    "----- HAZARD -----"
  );


  Serial.print(
    "Object    : "
  );

  Serial.println(
    hazard.object
  );


  Serial.print(
    "Direction : "
  );

  Serial.println(
    hazard.direction
  );


  Serial.print(
    "Risk      : "
  );

  Serial.println(
    hazard.risk
  );


  Serial.print(
    "TTC       : "
  );


  if (hazard.ttcValid) {

    Serial.println(
      hazard.ttc
    );

  } else {

    Serial.println(
      "None"
    );
  }


  // OLED 표시
  showHazard(
    hazard
  );


  // 모터 작동
  controlMotor(
    hazard
  );


  // Pi에 ACK
  sendAck();
}
