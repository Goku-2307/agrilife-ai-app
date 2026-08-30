/*
  Freshroute - ESP32 Cold-Chain Environmental Telemetry System
  ==============================================================
  Reads Temperature & Relative Humidity from DHT22 / DHT11 sensor
  and transmits data via:
    1. USB Serial COM port (JSON string)
    2. Bluetooth Low Energy (BLE) Environmental Sensing Service
    3. WiFi HTTP POST (Optional)

  Wiring:
    ESP32 Pin 3.3V / 5V -> Sensor VCC
    ESP32 Pin GND       -> Sensor GND
    ESP32 Pin GPIO 4    -> Sensor DATA (with 10k pullup resistor)
*/

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "DHT.h"

// Define DHT Pin and Type
#define DHTPIN 4
#define DHTTYPE DHT22   // Change to DHT11 if using DHT11

DHT dht(DHTPIN, DHTTYPE);

// BLE UUIDs for Environmental Sensing Service (ESS)
#define SERVICE_UUID        "0000181A-0000-1000-8000-00805F9B34FB"
#define TEMP_CHAR_UUID      "00002A6E-0000-1000-8000-00805F9B34FB"
#define HUM_CHAR_UUID       "00002A6F-0000-1000-8000-00805F9B34FB"

BLEServer* pServer = NULL;
BLECharacteristic* pTempCharacteristic = NULL;
BLECharacteristic* pHumCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("[BLE] Mobile Phone Connected.");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("[BLE] Mobile Phone Disconnected.");
    }
};

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[ESP32] Initializing Cold-Chain Sensor System...");

  // Initialize DHT Sensor
  dht.begin();
  Serial.println("[DHT] Sensor initialized.");

  // Initialize BLE Device
  BLEDevice::init("Freshroute-ESP32-Truck");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Temperature Characteristic (Notify)
  pTempCharacteristic = pService->createCharacteristic(
                      TEMP_CHAR_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pTempCharacteristic->addDescriptor(new BLE2902());

  // Humidity Characteristic (Notify)
  pHumCharacteristic = pService->createCharacteristic(
                      HUM_CHAR_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pHumCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  // Start BLE Advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("[BLE] Advertising started. Ready for mobile phone / dashboard connection.");
}

void loop() {
  // Read sensor values (takes ~250ms)
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature(); // Celsius

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("{\"error\": \"Failed to read from DHT sensor\"}");
    delay(2000);
    return;
  }

  // 1. Output clean JSON over USB Serial for the Dashboard
  // Format: {"temperature": 24.5, "humidity": 71.8}
  Serial.print("{\"temperature\": ");
  Serial.print(temperature, 2);
  Serial.print(", \"humidity\": ");
  Serial.print(humidity, 2);
  Serial.println("}");

  // 2. Notify over BLE if phone is connected
  if (deviceConnected) {
    // BLE Temperature format: 0.01 deg Celsius (int16)
    int16_t bleTemp = (int16_t)(temperature * 100);
    pTempCharacteristic->setValue((uint8_t*)&bleTemp, 2);
    pTempCharacteristic->notify();

    // BLE Humidity format: 0.01 % RH (uint16)
    uint16_t bleHum = (uint16_t)(humidity * 100);
    pHumCharacteristic->setValue((uint8_t*)&bleHum, 2);
    pHumCharacteristic->notify();
  }

  // Disconnecting reconnect handling
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("[BLE] Restarted advertising.");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  // Sample interval (e.g. 2000 ms)
  delay(2000);
}
