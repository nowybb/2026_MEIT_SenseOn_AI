#include "config.h"

#include "protocol.h"

#include "motor_control.h"

#include "display_control.h"

#include "ble_handler.h"


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(
    115200
  );


  delay(
    1000
  );


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


  // 모터
  setupMotor();


  // OLED
  setupDisplay();


  // BLE
  setupBLE();


  Serial.println(
    "[SYSTEM] READY"
  );
}


// =====================================================
// LOOP
// =====================================================

void loop() {

  // ===================================================
  // 모터 최소 유지시간 관리
  // ===================================================

  updateMotorState();


  // ===================================================
  // OLED 최소 표시시간 관리
  // ===================================================

  updateDisplayState();


  // ===================================================
  // BLE 연결이 끊기면 모터 OFF
  // ===================================================

  if (
    !isPiConnected() &&
    isMotorActive()
  ) {

    Serial.println(
      "[SAFETY] BLE disconnected -> BOTH OFF"
    );


    stopMotors();
  }


  // ===================================================
  // 새 패킷 없으면 반복
  // ===================================================

  if (
    !hasNewPacket()
  ) {

    delay(5);

    return;
  }


  // ===================================================
  // BLE 패킷
  // ===================================================

  String packet =
    getLatestPacket();


  Serial.println();

  Serial.println(
    "=============================="
  );


  Serial.print(
    "[PACKET] "
  );

  Serial.println(
    packet
  );


  // ===================================================
  // 파싱
  // ===================================================

  HazardData hazard =
    parsePacket(
      packet
    );


  if (
    !hazard.valid
  ) {

    Serial.println(
      "[ERROR] Invalid packet"
    );

    return;
  }


  // ===================================================
  // 정보 출력
  // ===================================================

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


  if (
    hazard.ttcValid
  ) {

    Serial.println(
      hazard.ttc
    );

  } else {

    Serial.println(
      "None"
    );
  }


  // ===================================================
  // OLED
  //
  // CAUTION = 최소 1초
  // DANGER = 최소 1.5초
  // ===================================================

  showHazard(
    hazard
  );


  // ===================================================
  // 모터
  //
  // CAUTION = 최소 1초
  // DANGER = 최소 1.5초
  // ===================================================

  controlMotor(
    hazard
  );


  // ===================================================
  // ACK
  // ===================================================

  sendAck();


  Serial.println(
    "[SYSTEM] Packet processing complete"
  );


  Serial.println(
    "=============================="
  );
}
