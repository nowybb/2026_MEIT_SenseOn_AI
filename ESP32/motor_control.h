#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>

#include "config.h"
#include "protocol.h"


// =====================================================
// 현재 모터 상태
// =====================================================

bool motorActive = false;


// 현재 실제 진동 상태
String activeRisk = "SAFE";


// 최소 진동 시작 시간
unsigned long holdStartTime = 0;


// 현재 최소 유지시간
unsigned long holdDuration = 0;


// 최소 진동시간 중
// SAFE가 한 번이라도 들어왔는지 기억
bool safeReceived = false;


// =====================================================
// 모터 초기화
// =====================================================

void setupMotor() {

  // A채널
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  // B채널
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);


  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);


  motorActive = false;

  activeRisk = "SAFE";

  safeReceived = false;


  Serial.println("[MOTOR] Ready");
}


// =====================================================
// 두 모터 모두 OFF
// =====================================================

void stopMotors() {

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);


  motorActive = false;

  activeRisk = "SAFE";

  safeReceived = false;


  Serial.println("[MOTOR] BOTH OFF");
}


// =====================================================
// 모터 작동 여부
// =====================================================

bool isMotorActive() {

  return motorActive;
}


// =====================================================
// 최소 진동시간 종료 여부
// =====================================================

bool isHoldFinished() {

  if (!motorActive) {
    return true;
  }


  return (
    millis() - holdStartTime
    >= holdDuration
  );
}


// =====================================================
// 두 모터 동시에 작동
// =====================================================

void runBothMotors(int power) {

  // A채널 모터
  analogWrite(
    RIGHT_IN1,
    power
  );

  analogWrite(
    RIGHT_IN2,
    0
  );


  // B채널 모터
  analogWrite(
    LEFT_IN1,
    power
  );

  analogWrite(
    LEFT_IN2,
    0
  );


  motorActive = true;


  Serial.print(
    "[MOTOR] BOTH PWM="
  );

  Serial.println(
    power
  );
}


// =====================================================
// 새 위험정보 처리
// =====================================================

void controlMotor(
  const HazardData& hazard
) {

  // ===================================================
  // SAFE
  // ===================================================

  if (
    hazard.risk == "SAFE"
  ) {

    Serial.println(
      "[MOTOR] SAFE received"
    );


    // 이미 꺼져 있으면 끝
    if (!motorActive) {

      return;
    }


    // SAFE가 한 번이라도 들어왔다는 사실 기억
    safeReceived = true;


    Serial.println(
      "[MOTOR] SAFE latched"
    );


    // 최소시간이 이미 끝났다면
    // 바로 OFF
    if (
      isHoldFinished()
    ) {

      Serial.println(
        "[MOTOR] Hold already finished -> OFF"
      );

      stopMotors();
    }


    // 최소시간이 안 끝났다면
    // 계속 진동
    return;
  }


  // ===================================================
  // DANGER
  // ===================================================

  if (
    hazard.risk == "DANGER"
  ) {

    // 이미 DANGER 상태면
    // 타이머를 계속 새로 시작하지 않음
    if (
      motorActive &&
      activeRisk == "DANGER"
    ) {

      return;
    }


    Serial.println(
      "[MOTOR] NEW DANGER"
    );


    activeRisk = "DANGER";


    // 새로운 DANGER이므로
    // 이전 SAFE 기록 제거
    safeReceived = false;


    // 두 모터 강하게
    runBothMotors(
      PWM_DANGER
    );


    // DANGER 최소 1500ms
    holdStartTime =
      millis();

    holdDuration =
      DANGER_HOLD_MS;


    Serial.println(
      "[MOTOR] DANGER hold = 1500ms"
    );


    return;
  }


  // ===================================================
  // CAUTION
  // ===================================================

  if (
    hazard.risk == "CAUTION"
  ) {

    // -----------------------------------------------
    // 현재 DANGER 최소시간이 아직 안 끝났으면
    // CAUTION으로 약하게 낮추지 않음
    // -----------------------------------------------

    if (
      motorActive &&
      activeRisk == "DANGER" &&
      !isHoldFinished()
    ) {

      Serial.println(
        "[MOTOR] DANGER hold active"
      );

      return;
    }


    // -----------------------------------------------
    // 이미 CAUTION 상태면
    // 타이머를 계속 새로 시작하지 않음
    // -----------------------------------------------

    if (
      motorActive &&
      activeRisk == "CAUTION"
    ) {

      return;
    }


    // -----------------------------------------------
    // 새 CAUTION
    // -----------------------------------------------

    Serial.println(
      "[MOTOR] NEW CAUTION"
    );


    activeRisk = "CAUTION";


    // 새 경고 시작
    safeReceived = false;


    // 두 모터 약하게
    runBothMotors(
      PWM_CAUTION
    );


    // CAUTION 최소 1000ms
    holdStartTime =
      millis();

    holdDuration =
      CAUTION_HOLD_MS;


    Serial.println(
      "[MOTOR] CAUTION hold = 1000ms"
    );


    return;
  }
}


// =====================================================
// loop()에서 계속 호출
//
// 최소시간이 끝났는지 계속 확인
// =====================================================

void updateMotorState() {

  // 모터가 꺼져 있으면 아무것도 안 함
  if (!motorActive) {

    return;
  }


  // 아직 최소 유지시간 중
  if (
    !isHoldFinished()
  ) {

    return;
  }


  // ===================================================
  // 최소시간 중 SAFE가 한 번이라도 있었다면
  // 시간 끝나는 순간 OFF
  // ===================================================

  if (
    safeReceived
  ) {

    Serial.println(
      "[MOTOR] Hold finished"
    );

    Serial.println(
      "[MOTOR] SAFE was detected -> BOTH OFF"
    );


    stopMotors();


    return;
  }


  // SAFE가 한 번도 없었다면
  // 현재 진동 계속 유지
}

#endif
