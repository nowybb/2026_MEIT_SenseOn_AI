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


  // 모터 초기화
  setupMotor();


  // OLED 초기화
  setupDisplay();


  // BLE 초기화
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
  // delay(1000), delay(1500)을 사용하지 않음
  // 진동 중에도 BLE 계속 수신
  // ===================================================

  updateMotorState();


  // ===================================================
  // BLE 연결이 끊기면
  // 안전을 위해 모터 즉시 OFF
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
  // 새 패킷이 없으면
  // 현재 상태 유지
  // ===================================================

  if (
    !hasNewPacket()
  ) {

    delay(5);

    return;
  }


  // ===================================================
  // BLE 패킷 가져오기
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
  // 받은 데이터 확인
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
  // 방향은 OLED에는 그대로 표시
  // 모터 방향 제어에는 사용하지 않음
  // ===================================================

  showHazard(
    hazard
  );


  // ===================================================
  // 두 모터 제어
  // ===================================================

  controlMotor(
    hazard
  );


  // ===================================================
  // Raspberry Pi에 ACK
  // ===================================================

  sendAck();


  Serial.println(
    "[SYSTEM] Packet processing complete"
  );


  Serial.println(
    "=============================="
  );
}
