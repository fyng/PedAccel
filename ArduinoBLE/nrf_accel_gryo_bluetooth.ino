#include <ArduinoBLE.h>           // Bluetooth Library
#include <Arduino_HTS221.h>       // Pressure Sensor Library
#include <Arduino_LPS22HB.h>      // Temperature Sensor Library
#include <Arduino_LSM9DS1.h>      // Magnetometer Sensor Library

// Initalizing global variables for sensor data to pass onto BLE
String a, g;

// BLE Service Name
BLEService customService("180C");

// BLE Characteristics
// Syntax: BLE<DATATYPE>Characteristic <NAME>(<UUID>, <PROPERTIES>, <DATA LENGTH>)
BLEStringCharacteristic ble_accel("2A56", BLERead | BLENotify, 31);
BLEStringCharacteristic ble_gyro("2A58", BLERead | BLENotify, 31);

// Function prototype
void readValues();

void setup()
{
    // Initalizing all the sensors
    IMU.begin();
    Serial.begin(9600);
    while (!Serial);
    if (!BLE.begin())
    {
        Serial.println("BLE failed to Initiate");
        delay(500);
        while (1);
    }
    // Setting BLE Name
    BLE.setLocalName("DT 6 BLE");
    
    // Setting BLE Service Advertisment
    BLE.setAdvertisedService(customService);
    
    // Adding characteristics to BLE Service Advertisment
    customService.addCharacteristic(ble_accel);
    customService.addCharacteristic(ble_gyro);

    // Adding the service to the BLE stack
    BLE.addService(customService);

    // Start advertising
    BLE.advertise();
    Serial.println("Bluetooth device is now active, waiting for connections...");
}

void loop()
{
    // Variable to check if cetral device is connected
    BLEDevice central = BLE.central();
    if (central)
    {
        Serial.print("Connected to central: ");
        Serial.println(central.address());
        while (central.connected())
        {
            delay(200);
            
            // Read values from sensors
            readValues();

            // Writing sensor values to the characteristic
            ble_accel.writeValue(a);
            ble_gyro.writeValue(g);

            // Displaying the sensor values on the Serial Monitor
            Serial.println("Reading Sensors");
            Serial.println(a);
            Serial.println(g);
            Serial.println("\n");
            delay(1000);
        }
    }
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
}

void readValues()
{
    // Reading raw sensor values from three sensors
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
       float x, y, z;
       float e , b, c;
      IMU.readAcceleration(x,y,z);
      IMU.readGyroscope(e,b,c);

          

    // Saving sensor values into a user presentable way with units
    a = "X:" + read_accel(x) + ", Y:" + read_accel(y) + ", Z:" + read_accel(z);
    g = "A:" + read_accel(e) + ", B:" + read_accel(b) + ", C:" + read_accel(c);
    }
}

String read_accel(float x) { 
    char buff[8];
    snprintf (buff, sizeof(buff), "%f", x);

    return String(buff);

  }

