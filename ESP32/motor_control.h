#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>

#include "config.h"
#include "protocol.h"


// =====================================================
// 현재 모터 상태
// =====================================================

bool motorActive = false;


// 현재 실제로 진동시키고 있는 위험도
String activeRisk = "SAFE";


// 현재 실제로 진동시키고 있는 방향
String activeDirection = "CENTER";


// 가장 최근에 AI에서 받은 상태
String latestRisk = "SAFE";


// 가장 최근에 AI에서 받은 방향
String latestDirection = "CENTER";


// 최소 진동 유지 시작 시간
unsigned long holdStartTime = 0;


// 현재 최소 유지시간
unsigned long holdDuration = 0;


// =====================================================
// 모터 초기화
// =====================================================

void setupMotor() {

  // A = 오른쪽 모터
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  // B = 왼쪽 모터
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
// 현재 모터가 켜져 있는지
// =====================================================

bool isMotorActive() {

  return motorActive;
}


// =====================================================
// 최소 유지시간이 끝났는지 확인
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

  // 먼저 양쪽 출력 초기화
  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);


  // ===================================================
  // RIGHT
  //
  // A채널 = 오른쪽 모터
  // ===================================================

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


  // ===================================================
  // LEFT
  //
  // B채널 = 왼쪽 모터
  // ===================================================

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


  // ===================================================
  // CENTER
  //
  // 양쪽 모터
  // ===================================================

  else if (direction == "CENTER") {

    // 오른쪽 A
    analogWrite(
      RIGHT_IN1,
      power
    );

    analogWrite(
      RIGHT_IN2,
      0
    );


    // 왼쪽 B
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
// Raspberry Pi에서 새 위험정보가 들어왔을 때 호출
// =====================================================

void controlMotor(
  const HazardData& hazard
) {

  // ---------------------------------------------------
  // 최신 AI 상태 저장
  // ---------------------------------------------------

  latestRisk =
    hazard.risk;

  latestDirection =
    hazard.direction;


  Serial.print(
    "[MOTOR] Latest risk = "
  );

  Serial.println(
    latestRisk
  );


  // ===================================================
  // DANGER
  //
  // 들어오는 즉시 강한 진동
  // 그리고 800ms 새로 시작
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


    // 즉시 강한 진동
    runMotor(
      hazard.direction,
      PWM_DANGER
    );


    // DANGER 최소시간 새로 시작
    holdStartTime =
      millis();

    holdDuration =
      DANGER_HOLD_MS;


    Serial.println(
      "[MOTOR] DANGER minimum hold = 800ms"
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


    // -------------------------------------------------
    // 현재 DANGER의 최소 800ms가 아직 안 끝난 경우
    //
    // CAUTION이 들어와도
    // DANGER를 바로 약하게 만들지 않음
    // -------------------------------------------------

    if (
      activeRisk == "DANGER" &&
      !isHoldFinished()
    ) {

      Serial.println(
        "[MOTOR] DANGER hold still active"
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


    // -------------------------------------------------
    // CAUTION 시작
    // -------------------------------------------------

    activeRisk =
      "CAUTION";

    activeDirection =
      hazard.direction;


    // 즉시 약한 진동
    runMotor(
      hazard.direction,
      PWM_CAUTION
    );


    // CAUTION 최소시간 시작
    holdStartTime =
      millis();

    holdDuration =
      CAUTION_HOLD_MS;


    Serial.println(
      "[MOTOR] CAUTION minimum hold = 500ms"
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


    // 이미 모터 OFF 상태
    if (!motorActive) {

      Serial.println(
        "[MOTOR] Already OFF"
      );

      return;
    }


    // -------------------------------------------------
    // 최소 유지시간이 이미 끝났다면
    // SAFE 수신 즉시 OFF
    // -------------------------------------------------

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


    // -------------------------------------------------
    // 아직 최소시간이 안 끝남
    //
    // SAFE가 들어와도 모터 계속 유지
    // -------------------------------------------------

    Serial.println(
      "[MOTOR] SAFE received, but minimum hold is active"
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
// loop()에서 계속 호출
//
// SAFE가 최소 유지시간 도중 들어왔을 경우
// 시간이 끝난 순간 실제로 모터를 OFF
// =====================================================

void updateMotorState() {

  // 모터가 꺼져 있으면 아무것도 안 함
  if (!motorActive) {
    return;
  }


  // 최소시간이 아직 안 끝남
  if (!isHoldFinished()) {
    return;
  }


  // ===================================================
  // 최소 유지시간이 끝났고
  // 가장 최근 AI 상태가 SAFE
  // ===================================================

  if (
    latestRisk == "SAFE"
  ) {

    Serial.println(
      "[MOTOR] Minimum hold finished"
    );

    Serial.println(
      "[MOTOR] Latest state = SAFE"
    );

    Serial.println(
      "[MOTOR] Motor OFF"
    );


    stopMotors();


    return;
  }


  // ===================================================
  // DANGER 유지시간이 끝났는데
  // 최신 상태가 CAUTION인 경우
  //
  // DANGER -> CAUTION으로 낮춤
  // ===================================================

  if (
    activeRisk == "DANGER" &&
    latestRisk == "CAUTION"
  ) {

    Serial.println(
      "[MOTOR] DANGER hold finished"
    );

    Serial.println(
      "[MOTOR] Latest state = CAUTION"
    );

    Serial.println(
      "[MOTOR] Change DANGER -> CAUTION"
    );


    activeRisk =
      "CAUTION";

    activeDirection =
      latestDirection;


    runMotor(
      latestDirection,
      PWM_CAUTION
    );


    // CAUTION 최소 500ms 새로 시작
    holdStartTime =
      millis();

    holdDuration =
      CAUTION_HOLD_MS;


    return;
  }


  // ===================================================
  // 최신 상태도 계속 위험 상태라면
  // 현재 진동 그대로 유지
  // ===================================================
}

#endif
