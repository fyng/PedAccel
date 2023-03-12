#include <ArduinoBLE.h>           // Bluetooth Library
#include <Arduino_LSM9DS1.h>      // Accel and Gyro Sensor Library
#define BLE_SENSE_UUID(val) ("6fbe1da7-" val "-44de-92c4-bb6e04fb0212")

// Initalizing global variables for sensor data to pass onto BLE


// BLE Service Name
BLEService service(BLE_SENSE_UUID("0000"));

// BLE Characteristics
// Syntax: BLE<DATATYPE>Characteristic <NAME>(<UUID>, <PROPERTIES>, <DATA LENGTH>)

//Float
BLECharacteristic ble_accel(BLE_SENSE_UUID("3001"),BLENotify, 3 * sizeof(float));
BLECharacteristic ble_gyro(BLE_SENSE_UUID("3002"), BLENotify, 3* sizeof(float));

// Function prototype
void readValues();

void setup()
{
  float x, y, z, g1, g2, g3;
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
    BLE.setAdvertisedService(service);
    
    // Adding characteristics to BLE Service Advertisment
    service.addCharacteristic(ble_accel);
    service.addCharacteristic(ble_gyro);

    // Adding the service to the BLE stack
    BLE.addService(service);

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
            // Reading raw sensor values from three sensors
          if ( IMU.accelerationAvailable()) {
               float x, y, z;
              IMU.readAcceleration(x, y, z);
              float acceleration[3] = { x, y, z };
              Serial.print("Reading Sensors");
              delay(1000);
              Serial.println();
              Serial.print("x"); 
              Serial.print(x);
              Serial.println();
              Serial.print("y");
              Serial.print(y);
              Serial.println();
              Serial.print("z");
              Serial.print(z);
              Serial.println();
              ble_accel.writeValue(acceleration, sizeof(acceleration));
    }
 
          if (IMU.gyroscopeAvailable()) {
             float x, y, z;
            IMU.readGyroscope(x, y, z);
            float dps[3] = { x, y, z };
            ble_gyro.writeValue(dps, sizeof(dps));
    }


          // Displaying the sensor values on the Serial Monitor
    }
 }
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
}

