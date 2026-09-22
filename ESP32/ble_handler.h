// ==============================
// ble_handler.h
// ==============================
#ifndef BLE_HANDLER_H
#define BLE_HANDLER_H

#include <Arduino.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "config.h"
#include "motor_control.h"


String latestPacket = "";

bool newPacketAvailable = false;

bool piConnected = false;

unsigned long lastPacketTime = 0;

BLECharacteristic*
  notifyCharacteristic = nullptr;


// BLE 연결 상태
class ServerCallbacks :
  public BLEServerCallbacks {

  void onConnect(
    BLEServer* server
  ) override {

    piConnected = true;

    Serial.println(
      "[BLE] Raspberry Pi connected"
    );
  }


  void onDisconnect(
    BLEServer* server
  ) override {

    piConnected = false;

    stopMotors();


    Serial.println(
      "[BLE] Raspberry Pi disconnected"
    );


    delay(100);


    server->startAdvertising();


    Serial.println(
      "[BLE] Advertising restarted"
    );
  }
};


// Pi -> ESP32 데이터 수신
class WriteCallbacks :
  public BLECharacteristicCallbacks {

  void onWrite(
    BLECharacteristic* characteristic
  ) override {

    String value =
      characteristic->getValue();


    if (value.length() == 0) {
      return;
    }


    latestPacket =
      value;


    newPacketAvailable =
      true;


    lastPacketTime =
      millis();


    Serial.print(
      "[BLE RX] "
    );

    Serial.println(
      latestPacket
    );
  }
};


// BLE 시작
void setupBLE() {

  BLEDevice::init(
    DEVICE_NAME
  );


  BLEServer* server =
    BLEDevice::createServer();


  server->setCallbacks(
    new ServerCallbacks()
  );


  BLEService* service =
    server->createService(
      SERVICE_UUID
    );


  BLECharacteristic* writeCharacteristic =
    service->createCharacteristic(

      WRITE_CHARACTERISTIC_UUID,

      BLECharacteristic::PROPERTY_WRITE
    );


  writeCharacteristic->setCallbacks(
    new WriteCallbacks()
  );


  notifyCharacteristic =
    service->createCharacteristic(

      NOTIFY_CHARACTERISTIC_UUID,

      BLECharacteristic::PROPERTY_NOTIFY
    );


  notifyCharacteristic->addDescriptor(
    new BLE2902()
  );


  service->start();


  BLEAdvertising* advertising =
    BLEDevice::getAdvertising();


  advertising->addServiceUUID(
    SERVICE_UUID
  );


  advertising->setScanResponse(
    true
  );


  advertising->start();


  Serial.println(
    "[BLE] SenseOn_ESP32 ready"
  );
}


// 새 패킷 있는지
bool hasNewPacket() {

  return newPacketAvailable;
}


// 패킷 가져오기
String getLatestPacket() {

  newPacketAvailable =
    false;

  return latestPacket;
}


// ACK 보내기
void sendAck() {

  if (
    !piConnected ||
    notifyCharacteristic == nullptr
  ) {

    return;
  }


  notifyCharacteristic->setValue(
    "ACK"
  );


  notifyCharacteristic->notify();


  Serial.println(
    "[BLE TX] ACK"
  );
}


// 연결 상태
bool isPiConnected() {

  return piConnected;
}


// 마지막 데이터 수신 시간
unsigned long getLastPacketTime() {

  return lastPacketTime;
}

#endif
