#ifndef BLE_HANDLER_H
#define BLE_HANDLER_H

#include <Arduino.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "config.h"
#include "motor_control.h"


// =====================================================
// BLE 상태값
// =====================================================

String latestPacket = "";

bool newPacketAvailable = false;
bool piConnected = false;

unsigned long lastPacketTime = 0;

BLEServer* bleServer = nullptr;

BLECharacteristic* writeCharacteristic = nullptr;
BLECharacteristic* notifyCharacteristic = nullptr;


// =====================================================
// BLE 연결 / 연결 해제 Callback
// =====================================================

class ServerCallbacks :
  public BLEServerCallbacks {

  void onConnect(
    BLEServer* server
  ) override {

    piConnected = true;

    Serial.println();
    Serial.println("================================");
    Serial.println("[BLE EVENT] CLIENT CONNECTED");
    Serial.println("[BLE] Raspberry Pi connected");
    Serial.println("================================");
  }


  void onDisconnect(
    BLEServer* server
  ) override {

    piConnected = false;

    Serial.println();
    Serial.println("================================");
    Serial.println("[BLE EVENT] CLIENT DISCONNECTED");
    Serial.println("[BLE] Raspberry Pi disconnected");
    Serial.println("================================");


    // 안전을 위해 모터 OFF
    Serial.println(
      "[BLE] Stopping motors..."
    );

    stopMotors();

    Serial.println(
      "[BLE] Motors stopped"
    );


    // 잠시 대기
    delay(300);


    // 다시 광고 시작
    Serial.println(
      "[BLE] Restarting advertising..."
    );

    server->startAdvertising();

    Serial.println(
      "[BLE] Advertising restarted"
    );

    Serial.println(
      "[BLE] Waiting for Raspberry Pi..."
    );
  }
};


// =====================================================
// Raspberry Pi -> ESP32 데이터 수신 Callback
// =====================================================

class WriteCallbacks :
  public BLECharacteristicCallbacks {

  void onWrite(
    BLECharacteristic* characteristic
  ) override {

    Serial.println();
    Serial.println(
      "---------- BLE WRITE ----------"
    );

    Serial.println(
      "[BLE RX] Write callback entered"
    );


    String value =
      characteristic->getValue();


    Serial.print(
      "[BLE RX] Received bytes: "
    );

    Serial.println(
      value.length()
    );


    if (value.length() == 0) {

      Serial.println(
        "[BLE RX] Empty packet"
      );

      Serial.println(
        "-------------------------------"
      );

      return;
    }


    latestPacket = value;

    newPacketAvailable = true;

    lastPacketTime = millis();


    Serial.print(
      "[BLE RX] Packet: "
    );

    Serial.println(
      latestPacket
    );


    Serial.print(
      "[BLE RX] millis: "
    );

    Serial.println(
      lastPacketTime
    );


    Serial.println(
      "[BLE RX] Packet stored successfully"
    );

    Serial.println(
      "-------------------------------"
    );
  }
};


// =====================================================
// BLE 초기화
// =====================================================

void setupBLE() {

  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "[BLE INIT] Starting BLE setup"
  );


  // -------------------------------------
  // Device 이름 설정
  // -------------------------------------

  Serial.print(
    "[BLE INIT] Device name: "
  );

  Serial.println(
    DEVICE_NAME
  );


  BLEDevice::init(
    DEVICE_NAME
  );


  Serial.println(
    "[BLE INIT] BLEDevice initialized"
  );


  // -------------------------------------
  // BLE Server 생성
  // -------------------------------------

  Serial.println(
    "[BLE INIT] Creating GATT server..."
  );


  bleServer =
    BLEDevice::createServer();


  if (bleServer == nullptr) {

    Serial.println(
      "[BLE ERROR] Server creation failed"
    );

    return;
  }


  Serial.println(
    "[BLE INIT] GATT server created"
  );


  bleServer->setCallbacks(
    new ServerCallbacks()
  );


  Serial.println(
    "[BLE INIT] Server callbacks registered"
  );


  // -------------------------------------
  // Service 생성
  // -------------------------------------

  Serial.println(
    "[BLE INIT] Creating service..."
  );


  Serial.print(
    "[BLE INIT] SERVICE UUID: "
  );

  Serial.println(
    SERVICE_UUID
  );


  BLEService* service =
    bleServer->createService(
      SERVICE_UUID
    );


  if (service == nullptr) {

    Serial.println(
      "[BLE ERROR] Service creation failed"
    );

    return;
  }


  Serial.println(
    "[BLE INIT] Service created"
  );


  // =====================================================
  // WRITE Characteristic
  // Pi -> ESP32
  // =====================================================

  Serial.println();
  Serial.println(
    "[BLE INIT] Creating WRITE characteristic"
  );


  Serial.print(
    "[BLE INIT] WRITE UUID: "
  );

  Serial.println(
    WRITE_CHARACTERISTIC_UUID
  );


  writeCharacteristic =
    service->createCharacteristic(

      WRITE_CHARACTERISTIC_UUID,

      BLECharacteristic::PROPERTY_WRITE
    );


  if (writeCharacteristic == nullptr) {

    Serial.println(
      "[BLE ERROR] WRITE characteristic creation failed"
    );

    return;
  }


  writeCharacteristic->setCallbacks(
    new WriteCallbacks()
  );


  Serial.println(
    "[BLE INIT] WRITE characteristic ready"
  );


  // =====================================================
  // NOTIFY Characteristic
  // ESP32 -> Pi
  // =====================================================

  Serial.println();
  Serial.println(
    "[BLE INIT] Creating NOTIFY characteristic"
  );


  Serial.print(
    "[BLE INIT] NOTIFY UUID: "
  );

  Serial.println(
    NOTIFY_CHARACTERISTIC_UUID
  );


  notifyCharacteristic =
    service->createCharacteristic(

      NOTIFY_CHARACTERISTIC_UUID,

      BLECharacteristic::PROPERTY_NOTIFY
    );


  if (notifyCharacteristic == nullptr) {

    Serial.println(
      "[BLE ERROR] NOTIFY characteristic creation failed"
    );

    return;
  }


  notifyCharacteristic->addDescriptor(
    new BLE2902()
  );


  Serial.println(
    "[BLE INIT] NOTIFY characteristic ready"
  );


  // -------------------------------------
  // Service 시작
  // -------------------------------------

  Serial.println();
  Serial.println(
    "[BLE INIT] Starting GATT service..."
  );


  service->start();


  Serial.println(
    "[BLE INIT] GATT service started"
  );


  // -------------------------------------
  // Advertising
  // -------------------------------------

  Serial.println(
    "[BLE INIT] Preparing advertising..."
  );


  BLEAdvertising* advertising =
    BLEDevice::getAdvertising();


  advertising->addServiceUUID(
    SERVICE_UUID
  );


  advertising->setScanResponse(
    true
  );


  Serial.println(
    "[BLE INIT] Starting advertising..."
  );


  advertising->start();


  Serial.println(
    "[BLE INIT] Advertising started"
  );


  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "[BLE] SenseOn_ESP32 READY"
  );

  Serial.println(
    "[BLE] Waiting for Raspberry Pi..."
  );

  Serial.println(
    "================================"
  );
}


// =====================================================
// 새 패킷 확인
// =====================================================

bool hasNewPacket() {

  return newPacketAvailable;
}


// =====================================================
// 최신 패킷 가져오기
// =====================================================

String getLatestPacket() {

  Serial.println(
    "[BLE] Main loop reading packet"
  );


  newPacketAvailable =
    false;


  return latestPacket;
}


// =====================================================
// ACK 전송
// =====================================================

void sendAck() {

  Serial.println(
    "[BLE TX] Preparing ACK..."
  );


  if (!piConnected) {

    Serial.println(
      "[BLE TX] ACK cancelled - no client connected"
    );

    return;
  }


  if (notifyCharacteristic == nullptr) {

    Serial.println(
      "[BLE TX] ACK cancelled - Notify characteristic NULL"
    );

    return;
  }


  notifyCharacteristic->setValue(
    "ACK"
  );


  Serial.println(
    "[BLE TX] ACK value set"
  );


  notifyCharacteristic->notify();


  Serial.println(
    "[BLE TX] ACK notification sent"
  );
}


// =====================================================
// 현재 Pi 연결 상태
// =====================================================

bool isPiConnected() {

  return piConnected;
}


// =====================================================
// 마지막 패킷 수신 시간
// =====================================================

unsigned long getLastPacketTime() {

  return lastPacketTime;
}

#endif
