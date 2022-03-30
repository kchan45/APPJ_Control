//Status reg 15:8 Error Code  7:3 Reserved 2:Boot Status 1:Boot Mode 0:busy

int read_reg(unsigned int reg)
{
  int reading = 0;
  set_reg(reg);

  Wire.requestFrom(ADDRESS, 2);

  reading = Wire.read();  // receive high byte (overwrites previous reading)
  Serial.println(reading);
  reading = reading << 8;    // shift high byte to be high 8 bits

  reading |= Wire.read(); // receive low byte as lower 8 bits
  Serial.print("reg:");
  Serial.print(reg);
  Serial.print("==0x");
  Serial.print(reading, HEX);
  Serial.print(" binary:");
  Serial.println(reading, BIN);
  return reading;
}
