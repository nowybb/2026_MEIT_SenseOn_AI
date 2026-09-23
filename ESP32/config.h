#ifndef CONFIG_H
#define CONFIG_H

// =====================================================
// BLE
// =====================================================

#define DEVICE_NAME "SenseOn_ESP32"

// BLE Service
#define SERVICE_UUID \
"9a1b0000-1111-2222-3333-444455556666"

// Raspberry Pi -> ESP32
#define WRITE_CHARACTERISTIC_UUID \
"9a1b1001-1111-2222-3333-444455556666"

// ESP32 -> Raspberry Pi
#define NOTIFY_CHARACTERISTIC_UUID \
"9a1b2002-1111-2222-3333-444455556666"


// =====================================================
// OLED
// =====================================================

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_ADDRESS 0x3C

#define OLED_SDA 21
#define OLED_SCL 22


// =====================================================
// DRV8833
//
// A = 오른쪽 모터
// B = 왼쪽 모터
// =====================================================

#define RIGHT_IN1 25
#define RIGHT_IN2 26

#define LEFT_IN1 27
#define LEFT_IN2 14


// =====================================================
// Motor PWM
// =====================================================

#define PWM_CAUTION 60
#define PWM_DANGER 150


// =====================================================
// Safety
// =====================================================

#define DATA_TIMEOUT_MS 1000

#endif
