void agc_enable(){
  byte error;
  Wire.beginTransmission(ADDRESS); // transmit to device #4
  Wire.write(0x01);
  Wire.write(0x05);
  Wire.write(0x00);
  Wire.write(0x01);

  error = Wire.endTransmission();    // stop transmitting
  if (error != 0){
    Serial.print("error=");
    Serial.println(error);
  }
}
