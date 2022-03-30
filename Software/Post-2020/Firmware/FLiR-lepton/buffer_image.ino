void buffer_image(void){
  //command to get from the lecture the ID of the frame
  int j = lepton_frame_packet[1];
  
  for (int i = 0; i < image_x; i++){
    //assign to the variable image[i][j] the data read from the lecture, starts assign from 4th and 5th positions due to the from [0]to[3] are the ID and the CRC
    image[i][j] = ((lepton_frame_packet[2 * i + 4] << 8) + lepton_frame_packet[2 * i + 5]); //-0x1000
  }
  
  if (j == 0x3B) {
    donecapturing = true;
  }
}
