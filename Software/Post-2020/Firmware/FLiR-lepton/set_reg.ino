//Status reg 15:8 Error Code  7:3 Reserved 2:Boot Status 1:Boot Mode 0:busy

void set_reg(unsigned int reg){
  byte error;
  Wire.beginTransmission(ADDRESS); // transmit to device #4
  Wire.write(reg >> 8 & 0xff);
  Wire.write(reg & 0xff);            // sends one byte

  error = Wire.endTransmission();    // stop transmitting
  if (error != 0){
    Serial.print("error=");
    Serial.println(error);
  }
}
