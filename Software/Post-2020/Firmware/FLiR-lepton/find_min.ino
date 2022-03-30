void find_min(void)
{
  for (int j = 0; j < image_y; j++){
    for (int i = 0; i < image_x; i++){

      if (image[i][j] < minval){
        minval = image[i][j]; 
      }
    }
  }
  Serial.println(minval);
}
