// ==============================
// motor_control.h
// ==============================
#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>

#include "config.h"
#include "protocol.h"


bool motorActive = false;


// 모터 초기화
void setupMotor() {

  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;

  Serial.println("[MOTOR] Ready");
}


// 모든 모터 OFF
void stopMotors() {

  analogWrite(RIGHT_IN1, 0);
  analogWrite(RIGHT_IN2, 0);

  analogWrite(LEFT_IN1, 0);
  analogWrite(LEFT_IN2, 0);

  motorActive = false;
}


// 모터 작동 여부
bool isMotorActive() {

  return motorActive;
}


// 실제 모터 제어
void controlMotor(
  const HazardData& hazard
) {

  stopMotors();


  // SAFE면 모터 끄기
  if (hazard.risk == "SAFE") {

    Serial.println(
      "[MOTOR] SAFE -> OFF"
    );

    return;
  }


  int power = 0;


  if (hazard.risk == "CAUTION") {

    power = PWM_CAUTION;

  } else if (hazard.risk == "DANGER") {

    power = PWM_DANGER;
  }


  // =========================
  // RIGHT
  // A채널
  // =========================

  if (hazard.direction == "RIGHT") {

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

    Serial.println(power);
  }


  // =========================
  // LEFT
  // B채널
  // =========================

  else if (hazard.direction == "LEFT") {

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

    Serial.println(power);
  }


  // =========================
  // CENTER
  // A + B 둘 다
  // =========================

  else if (hazard.direction == "CENTER") {

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

    Serial.println(power);
  }


  motorActive = true;
}

#endif
