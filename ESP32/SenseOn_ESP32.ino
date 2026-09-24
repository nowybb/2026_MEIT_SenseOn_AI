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


  // ===================================================
  // 모터 초기화
  // ===================================================

  setupMotor();


  // ===================================================
  // OLED 초기화
  // ===================================================

  setupDisplay();


  // ===================================================
  // BLE 초기화
  // ===================================================

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
  // 최소 진동 유지시간 관리
  //
  // delay(500), delay(800)을 사용하지 않음
  //
  // 그래서 진동 중에도 BLE 데이터 계속 수신 가능
  // ===================================================

  updateMotorState();


  // ===================================================
  // BLE 연결 자체가 끊어진 경우
  //
  // 안전을 위해 즉시 모터 OFF
  // ===================================================

  if (
    !isPiConnected() &&
    isMotorActive()
  ) {

    Serial.println(
      "[SAFETY] BLE disconnected -> Motor OFF"
    );


    stopMotors();
  }


  // ===================================================
  // 새 패킷이 없으면
  //
  // BLE 연결과 현재 모터 상태는 그대로 유지
  // ===================================================

  if (
    !hasNewPacket()
  ) {

    delay(5);

    return;
  }


  // ===================================================
  // 새 BLE 패킷 가져오기
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


  // 잘못된 패킷
  if (
    !hazard.valid
  ) {

    Serial.println(
      "[ERROR] Invalid packet"
    );


    Serial.println(
      "=============================="
    );


    return;
  }


  // ===================================================
  // 받은 위험 정보 출력
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
  // OLED에는 AI의 가장 최신 상태를 바로 표시
  // ===================================================

  Serial.println(
    "[SYSTEM] OLED update"
  );


  showHazard(
    hazard
  );


  // ===================================================
  // 모터 제어
  //
  // CAUTION
  // -> 최소 500ms
  //
  // DANGER
  // -> 최소 800ms
  //
  // SAFE
  // -> 최소시간 종료 후 OFF
  // ===================================================

  Serial.println(
    "[SYSTEM] Motor control"
  );


  controlMotor(
    hazard
  );


  // ===================================================
  // ACK
  //
  // BLE 연결은 계속 유지
  // ===================================================

  Serial.println(
    "[SYSTEM] Sending ACK"
  );


  sendAck();


  Serial.println(
    "[SYSTEM] Packet processing complete"
  );


  Serial.println(
    "=============================="
  );
}
