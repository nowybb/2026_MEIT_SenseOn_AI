#include "config.h"

#include "protocol.h"

#include "motor_control.h"

#include "display_control.h"

#include "ble_handler.h"


// =====================================================
// SETUP
// =====================================================

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

  // -------------------------------------
  // BLE 연결이 끊어졌으면
  // 안전을 위해 모터 OFF
  // -------------------------------------

  if (
    !isPiConnected() &&
    isMotorActive()
  ) {

    stopMotors();


    Serial.println(
      "[SAFETY] BLE disconnected -> OFF"
    );
  }


  // -------------------------------------
  // 새 BLE 데이터가 없으면
  // 현재 모터 상태 그대로 유지
  // -------------------------------------

  if (!hasNewPacket()) {

    delay(5);

    return;
  }


  // -------------------------------------
  // 새 패킷 가져오기
  // -------------------------------------

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


  // -------------------------------------
  // 패킷 파싱
  // -------------------------------------

  HazardData hazard =
    parsePacket(packet);


  // 잘못된 패킷
  if (!hazard.valid) {

    Serial.println(
      "[ERROR] Invalid packet"
    );

    Serial.println(
      "=============================="
    );

    return;
  }


  // -------------------------------------
  // 받은 정보 출력
  // -------------------------------------

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


  // -------------------------------------
  // OLED 업데이트
  // -------------------------------------

  Serial.println(
    "[SYSTEM] OLED update"
  );

  showHazard(
    hazard
  );


  // -------------------------------------
  // 모터 제어
  //
  // SAFE
  // -> 모터 OFF
  //
  // CAUTION / DANGER
  // -> 다음 SAFE가 올 때까지 계속 유지
  // -------------------------------------

  Serial.println(
    "[SYSTEM] Motor control"
  );

  controlMotor(
    hazard
  );


  // -------------------------------------
  // 처리 완료 ACK
  // -------------------------------------

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
