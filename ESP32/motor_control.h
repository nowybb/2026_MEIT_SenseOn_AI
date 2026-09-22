#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>
#include "config.h"
#include "protocol.h"

// 현재 모터가 작동 중인지 저장
bool motorActive = false;


// ==============================
// 모터 초기화
// ==============================
void setupMotor() {

  // A 채널 = 오른쪽 모터
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  // B 채널 = 왼쪽 모터
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);

  // 시작할 때 모터 OFF
  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;

  Serial.println("[MOTOR] Ready");
}


// ==============================
// 모든 모터 정지
// ==============================
void stopMotors() {

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;
}


// ==============================
// 모터 작동 상태 확인
// ==============================
bool isMotorActive() {

  return motorActive;
}


// ==============================
// 위험 정보에 따라 모터 제어
// ==============================
void controlMotor(const HazardData& hazard) {

  // 이전 진동 정지
  stopMotors();


  // SAFE면 진동 없음
  if (hazard.risk == "SAFE") {

    Serial.println("[MOTOR] SAFE -> OFF");

    return;
  }


  // 위험도에 따른 진동 세기
  int power = 0;

  if (hazard.risk == "CAUTION") {

    power = PWM_CAUTION;   // 60

  }
  else if (hazard.risk == "DANGER") {

    power = PWM_DANGER;    // 150
  }


  // ==============================
  // RIGHT
  // A 채널 = 오른쪽 모터
  // ==============================
  if (hazard.direction == "RIGHT") {

    analogWrite(RIGHT_IN1, power);
    analogWrite(RIGHT_IN2, 0);

    Serial.print("[MOTOR] RIGHT(A) PWM=");
    Serial.println(power);
  }


  // ==============================
  // LEFT
  // B 채널 = 왼쪽 모터
  // ==============================
  else if (hazard.direction == "LEFT") {

    analogWrite(LEFT_IN1, power);
    analogWrite(LEFT_IN2, 0);

    Serial.print("[MOTOR] LEFT(B) PWM=");
    Serial.println(power);
  }


  // ==============================
  // CENTER
  // 양쪽 모터
  // ==============================
  else if (hazard.direction == "CENTER") {

    // 오른쪽 A
    analogWrite(RIGHT_IN1, power);
    analogWrite(RIGHT_IN2, 0);

    // 왼쪽 B
    analogWrite(LEFT_IN1, power);
    analogWrite(LEFT_IN2, 0);

    Serial.print("[MOTOR] CENTER(A+B) PWM=");
    Serial.println(power);
  }


  motorActive = true;
}

#endif
