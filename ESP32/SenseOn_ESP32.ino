#include "config.h"
#include "protocol.h"
#include "motor_control.h"
#include "ble_handler.h"


bool motorActive = false;


void setup() {

    Serial.begin(115200);

    delay(1000);


    Serial.println();
    Serial.println("====================");
    Serial.println("   SenseOn ESP32");
    Serial.println("====================");


    setupMotor();

    setupBLE();


    Serial.println("[SYSTEM] READY");
}


void loop() {

    // =====================================
    // BLE 연결 끊기면 모터 OFF
    // =====================================

    if (!piConnected && motorActive) {

        stopMotors();

        motorActive = false;

        Serial.println(
            "[SAFETY] BLE disconnected -> motor OFF"
        );
    }


    // =====================================
    // 일정 시간 데이터 없으면 모터 OFF
    // =====================================

    if (
        motorActive &&
        millis() - lastPacketTime > DATA_TIMEOUT_MS
    ) {

        stopMotors();

        motorActive = false;

        Serial.println(
            "[SAFETY] Data timeout -> motor OFF"
        );
    }


    // 새 패킷이 없으면 끝
    if (!hasNewPacket()) {

        delay(5);

        return;
    }


    // =====================================
    // 1. Pi에서 패킷 수신
    // =====================================

    String packet =
        getLatestPacket();


    Serial.print("[PACKET] ");
    Serial.println(packet);


    // =====================================
    // 2. 패킷 분석
    // =====================================

    HazardData hazard =
        parsePacket(packet);


    if (!hazard.valid) {

        Serial.println(
            "[ERROR] Invalid packet"
        );

        return;
    }


    // =====================================
    // 3. 내용 확인
    // =====================================

    Serial.print("Object : ");
    Serial.println(hazard.object);

    Serial.print("Direction : ");
    Serial.println(hazard.direction);

    Serial.print("Risk : ");
    Serial.println(hazard.risk);


    if (hazard.ttcValid) {

        Serial.print("TTC : ");
        Serial.println(hazard.ttc);

    } else {

        Serial.println("TTC : None");
    }


    // =====================================
    // 4. 모터 제어
    // =====================================

    controlMotor(hazard);


    motorActive =
        hazard.risk != "SAFE";


    // =====================================
    // 5. ACK
    //
    // actuator 명령 적용 직후
    // Pi에게 ACK
    // =====================================

    sendAck();
}
