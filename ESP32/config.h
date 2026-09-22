// ==============================
// config.h
// ==============================
#ifndef CONFIG_H
#define CONFIG_H

// BLE
#define DEVICE_NAME "SenseOn_ESP32"

#define SERVICE_UUID \
"7f4d1000-2e3b-4f7a-9b2a-1c2d3e4f5001"

#define WRITE_CHARACTERISTIC_UUID \
"7f4d1001-2e3b-4f7a-9b2a-1c2d3e4f5001"

#define NOTIFY_CHARACTERISTIC_UUID \
"7f4d1002-2e3b-4f7a-9b2a-1c2d3e4f5001"


// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

#define OLED_SDA 21
#define OLED_SCL 22


// DRV8833
// A = 오른쪽 모터
// B = 왼쪽 모터

#define RIGHT_IN1 25   // AN1
#define RIGHT_IN2 26   // AN2

#define LEFT_IN1 27    // BN1
#define LEFT_IN2 14    // BN2


// 진동 세기
#define PWM_CAUTION 60
#define PWM_DANGER 150


// 데이터 끊겼을 때 안전 정지
#define DATA_TIMEOUT_MS 1000

#endif
