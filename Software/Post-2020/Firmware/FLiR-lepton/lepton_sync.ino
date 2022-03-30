void lepton_sync(void){
  int i;
  int data = 0x0f;

  digitalWriteDirect(10, HIGH);
  delay(185);
  
  while (data & 0x0f == 0x0f){
    digitalWriteDirect(10, LOW);
    data = SPI.transfer(0x00) << 8;
    data |= SPI.transfer(0x00);
    digitalWriteDirect(10, HIGH);

    for (i = 0; i < ((VOSPI_FRAME_SIZE - 2) / 2); i++){
      digitalWriteDirect(10, LOW);
      SPI.transfer(0x00);
      SPI.transfer(0x00);
      digitalWriteDirect(10, HIGH);
    }
  }
}
