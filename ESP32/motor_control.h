#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <Arduino.h>

#include "config.h"
#include "protocol.h"


void stopMotors() {

    analogWrite(LEFT_IN1, 0);
    analogWrite(LEFT_IN2, 0);

    analogWrite(RIGHT_IN1, 0);
    analogWrite(RIGHT_IN2, 0);
}


void setupMotor() {

    pinMode(LEFT_IN1, OUTPUT);
    pinMode(LEFT_IN2, OUTPUT);

    pinMode(RIGHT_IN1, OUTPUT);
    pinMode(RIGHT_IN2, OUTPUT);

    stopMotors();
}


// 위험 정보에 따라 모터 제어
void controlMotor(const HazardData& hazard) {

    // 이전 상태 초기화
    stopMotors();


    // SAFE면 진동 없음
    if (hazard.risk == "SAFE") {

        Serial.println("[MOTOR] SAFE -> OFF");

        return;
    }


    int pwmValue;


    if (hazard.risk == "CAUTION") {

        pwmValue = PWM_CAUTION;

    } else {

        pwmValue = PWM_DANGER;
    }


    // 왼쪽 위험
    if (hazard.direction == "LEFT") {

        analogWrite(LEFT_IN1, pwmValue);
        analogWrite(LEFT_IN2, 0);

        Serial.println("[MOTOR] LEFT");
    }


    // 오른쪽 위험
    else if (hazard.direction == "RIGHT") {

        analogWrite(RIGHT_IN1, pwmValue);
        analogWrite(RIGHT_IN2, 0);

        Serial.println("[MOTOR] RIGHT");
    }


    // 중앙 위험
    else if (hazard.direction == "CENTER") {

        analogWrite(LEFT_IN1, pwmValue);
        analogWrite(LEFT_IN2, 0);

        analogWrite(RIGHT_IN1, pwmValue);
        analogWrite(RIGHT_IN2, 0);

        Serial.println("[MOTOR] CENTER -> BOTH");
    }


    Serial.print("[MOTOR] PWM = ");
    Serial.println(pwmValue);
}

#endif
