int read_data(){
  int i;
  int data;
  int payload_length;

  while (read_reg(0x2) & 0x01){
    Serial.println("busy");
  }

  payload_length = read_reg(0x6);
  Serial.print("payload_length=");
  Serial.println(payload_length);

  Wire.requestFrom(ADDRESS, payload_length);
  
  for (i = 0; i < (payload_length / 2); i++)
  {
    data = Wire.read() << 8;
    data |= Wire.read();
    Serial.println(data, DEC);
  }
return data;
}
