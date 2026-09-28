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
  // 최소 진동시간 관리
  //
  // delay(700), delay(1000)을 쓰지 않아서
  // 진동 중에도 BLE 패킷 계속 받을 수 있음
  // ===================================================

  updateMotorState();


  // ===================================================
  // BLE 연결이 끊어진 경우
  // 안전을 위해 즉시 모터 OFF
  // ===================================================

  if (
    !isPiConnected() &&
    isMotorActive()
  ) {

    Serial.println(
      "[SAFETY] BLE disconnected -> OFF"
    );

    stopMotors();
  }


  // ===================================================
  // 새 패킷이 없으면
  // 현재 진동 상태 유지
  // ===================================================

  if (
    !hasNewPacket()
  ) {

    delay(5);

    return;
  }


  // ===================================================
  // 패킷 가져오기
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
  // 패킷 파싱
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
  // OLED에는 최신 AI 상태 표시
  // ===================================================

  showHazard(
    hazard
  );


  // ===================================================
  // 모터 제어
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
