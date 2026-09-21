#ifndef BLE_HANDLER_H
#define BLE_HANDLER_H

#include <Arduino.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "config.h"


String latestPacket = "";

bool newPacketAvailable = false;
bool piConnected = false;

unsigned long lastPacketTime = 0;


BLECharacteristic* notifyCharacteristic = nullptr;

BLEServer* bleServer = nullptr;


// 연결 / 연결 해제
class ServerCallbacks : public BLEServerCallbacks {

    void onConnect(BLEServer* server) override {

        piConnected = true;

        Serial.println("[BLE] Raspberry Pi connected");
    }


    void onDisconnect(BLEServer* server) override {

        piConnected = false;

        Serial.println("[BLE] Raspberry Pi disconnected");

        delay(100);

        server->startAdvertising();

        Serial.println("[BLE] Advertising restarted");
    }
};


// Pi -> ESP32 데이터 수신
class WriteCallbacks : public BLECharacteristicCallbacks {

    void onWrite(BLECharacteristic* characteristic) override {

        String value =
            characteristic->getValue();


        if (value.length() == 0) {
            return;
        }


        latestPacket = value;

        newPacketAvailable = true;

        lastPacketTime = millis();


        Serial.print("[BLE RX] ");
        Serial.println(latestPacket);
    }
};


void setupBLE() {

    BLEDevice::init(
        DEVICE_NAME
    );


    bleServer =
        BLEDevice::createServer();


    bleServer->setCallbacks(
        new ServerCallbacks()
    );


    BLEService* service =
        bleServer->createService(
            SERVICE_UUID
        );


    // Pi -> ESP32
    BLECharacteristic* writeCharacteristic =
        service->createCharacteristic(

            WRITE_CHARACTERISTIC_UUID,

            BLECharacteristic::PROPERTY_WRITE
        );


    writeCharacteristic->setCallbacks(
        new WriteCallbacks()
    );


    // ESP32 -> Pi ACK
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


    advertising->setScanResponse(true);

    advertising->start();


    Serial.println("[BLE] Ready");
}


bool hasNewPacket() {

    return newPacketAvailable;
}


String getLatestPacket() {

    newPacketAvailable = false;

    return latestPacket;
}


// Pi가 기다리는 ACK
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


    Serial.println("[ACK TX] ACK");
}

#endif
