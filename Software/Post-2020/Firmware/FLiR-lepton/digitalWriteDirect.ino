inline void digitalWriteDirect(int pin, boolean val) {
  if (val) {
    g_APinDescription[pin].pPort -> PIO_SODR = g_APinDescription[pin].ulPin;
  }else{
    g_APinDescription[pin].pPort -> PIO_CODR = g_APinDescription[pin].ulPin;
  }
}
