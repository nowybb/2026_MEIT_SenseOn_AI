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


// DRV8833
#define LEFT_IN1 25
#define LEFT_IN2 26

#define RIGHT_IN1 27
#define RIGHT_IN2 14


// 진동 세기
#define PWM_CAUTION 102   // 약 40%
#define PWM_DANGER 230    // 약 90%


// 일정 시간 새 데이터 없으면 모터 OFF
#define DATA_TIMEOUT_MS 1000

#endif
