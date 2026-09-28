#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>

#include "config.h"
#include "protocol.h"


// =====================================================
// 현재 상태
// =====================================================

bool motorActive = false;

// 실제로 현재 진동 중인 위험도
String activeRisk = "SAFE";

// 실제로 현재 진동 중인 방향
String activeDirection = "CENTER";

// 가장 최근 AI 상태
String latestRisk = "SAFE";

// 가장 최근 AI 방향
String latestDirection = "CENTER";

// 최소 진동 시작 시간
unsigned long holdStartTime = 0;

// 현재 최소 유지시간
unsigned long holdDuration = 0;


// =====================================================
// 모터 초기화
// =====================================================

void setupMotor() {

  // A = 오른쪽
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  // B = 왼쪽
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;

  activeRisk = "SAFE";
  latestRisk = "SAFE";

  Serial.println("[MOTOR] Ready");
}


// =====================================================
// 모든 모터 OFF
// =====================================================

void stopMotors() {

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;

  activeRisk = "SAFE";

  Serial.println("[MOTOR] OFF");
}


// =====================================================
// 모터가 켜져 있는지
// =====================================================

bool isMotorActive() {

  return motorActive;
}


// =====================================================
// 최소 유지시간 끝났는지
// =====================================================

bool isHoldFinished() {

  if (!motorActive) {
    return true;
  }

  unsigned long elapsed =
    millis() - holdStartTime;

  return (
    elapsed >= holdDuration
  );
}


// =====================================================
// 남은 최소 유지시간
// =====================================================

unsigned long getRemainingHoldTime() {

  if (!motorActive) {
    return 0;
  }

  unsigned long elapsed =
    millis() - holdStartTime;

  if (elapsed >= holdDuration) {
    return 0;
  }

  return (
    holdDuration - elapsed
  );
}


// =====================================================
// 실제 모터 출력
// =====================================================

void runMotor(
  String direction,
  int power
) {

  // 기존 출력 초기화
  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);


  // ==============================
  // RIGHT
  // A채널
  // ==============================

  if (direction == "RIGHT") {

    analogWrite(
      RIGHT_IN1,
      power
    );

    analogWrite(
      RIGHT_IN2,
      0
    );

    Serial.print(
      "[MOTOR] RIGHT(A) PWM="
    );

    Serial.println(
      power
    );
  }


  // ==============================
  // LEFT
  // B채널
  // ==============================

  else if (direction == "LEFT") {

    analogWrite(
      LEFT_IN1,
      power
    );

    analogWrite(
      LEFT_IN2,
      0
    );

    Serial.print(
      "[MOTOR] LEFT(B) PWM="
    );

    Serial.println(
      power
    );
  }


  // ==============================
  // CENTER
  // 양쪽
  // ==============================

  else if (direction == "CENTER") {

    analogWrite(
      RIGHT_IN1,
      power
    );

    analogWrite(
      RIGHT_IN2,
      0
    );

    analogWrite(
      LEFT_IN1,
      power
    );

    analogWrite(
      LEFT_IN2,
      0
    );

    Serial.print(
      "[MOTOR] CENTER(A+B) PWM="
    );

    Serial.println(
      power
    );
  }


  motorActive = true;
}


// =====================================================
// 새 AI 결과 수신
// =====================================================

void controlMotor(
  const HazardData& hazard
) {

  // 최신 AI 상태 저장
  latestRisk =
    hazard.risk;

  latestDirection =
    hazard.direction;


  // ===================================================
  // DANGER
  //
  // 즉시 강하게
  // 최소 1000ms 새로 시작
  // ===================================================

  if (
    hazard.risk == "DANGER"
  ) {

    Serial.println(
      "[MOTOR] DANGER received"
    );

    activeRisk =
      "DANGER";

    activeDirection =
      hazard.direction;


    runMotor(
      hazard.direction,
      PWM_DANGER
    );


    holdStartTime =
      millis();

    holdDuration =
      DANGER_HOLD_MS;


    Serial.println(
      "[MOTOR] DANGER minimum hold = 1000ms"
    );

    return;
  }


  // ===================================================
  // CAUTION
  // ===================================================

  if (
    hazard.risk == "CAUTION"
  ) {

    Serial.println(
      "[MOTOR] CAUTION received"
    );


    // DANGER 최소시간이 아직 안 끝났으면
    // CAUTION으로 바로 낮추지 않음
    if (
      activeRisk == "DANGER" &&
      !isHoldFinished()
    ) {

      Serial.println(
        "[MOTOR] DANGER hold active"
      );

      Serial.print(
        "[MOTOR] Remaining = "
      );

      Serial.print(
        getRemainingHoldTime()
      );

      Serial.println(
        " ms"
      );

      return;
    }


    activeRisk =
      "CAUTION";

    activeDirection =
      hazard.direction;


    runMotor(
      hazard.direction,
      PWM_CAUTION
    );


    holdStartTime =
      millis();

    holdDuration =
      CAUTION_HOLD_MS;


    Serial.println(
      "[MOTOR] CAUTION minimum hold = 700ms"
    );

    return;
  }


  // ===================================================
  // SAFE
  // ===================================================

  if (
    hazard.risk == "SAFE"
  ) {

    Serial.println(
      "[MOTOR] SAFE received"
    );


    // 이미 꺼져 있음
    if (!motorActive) {

      Serial.println(
        "[MOTOR] Already OFF"
      );

      return;
    }


    // 최소 유지시간이 끝난 상태면 즉시 OFF
    if (
      isHoldFinished()
    ) {

      Serial.println(
        "[MOTOR] Minimum hold finished"
      );

      Serial.println(
        "[MOTOR] SAFE -> OFF"
      );

      stopMotors();

      return;
    }


    // 최소시간이 아직 안 끝났으면 계속 진동
    Serial.println(
      "[MOTOR] SAFE received but hold active"
    );

    Serial.print(
      "[MOTOR] Remaining = "
    );

    Serial.print(
      getRemainingHoldTime()
    );

    Serial.println(
      " ms"
    );

    return;
  }
}


// =====================================================
// loop에서 계속 호출
// 최소시간 끝난 뒤 최신 상태 확인
// =====================================================

void updateMotorState() {

  if (!motorActive) {
    return;
  }


  if (!isHoldFinished()) {
    return;
  }


  // ==============================
  // 최신 상태가 SAFE
  // -> OFF
  // ==============================

  if (
    latestRisk == "SAFE"
  ) {

    Serial.println(
      "[MOTOR] Hold finished"
    );

    Serial.println(
      "[MOTOR] Latest SAFE -> OFF"
    );

    stopMotors();

    return;
  }


  // ==============================
  // DANGER 유지시간 끝났는데
  // 최신 상태가 CAUTION
  // -> CAUTION으로 낮춤
  // ==============================

  if (
    activeRisk == "DANGER" &&
    latestRisk == "CAUTION"
  ) {

    Serial.println(
      "[MOTOR] DANGER hold finished"
    );

    Serial.println(
      "[MOTOR] Change to CAUTION"
    );


    activeRisk =
      "CAUTION";

    activeDirection =
      latestDirection;


    runMotor(
      latestDirection,
      PWM_CAUTION
    );


    holdStartTime =
      millis();

    holdDuration =
      CAUTION_HOLD_MS;

    return;
  }


  // 최신 상태가 계속 위험이면
  // 진동 그대로 유지
}

#endif
