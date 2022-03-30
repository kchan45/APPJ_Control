 void read_lepton_frame(void){
  int i;
  for (i = 0; i < (VOSPI_FRAME_SIZE / 2); i++){
    digitalWriteDirect(10, LOW);
    //  send in the address and value via SPI:
    lepton_frame_packet[2 * i] = SPI.transfer(0x00);
    lepton_frame_packet[2 * i + 1] = SPI.transfer(0x00);
    // take the SS pin high to de-select the chip:
    digitalWriteDirect(10, HIGH);
  }
}
