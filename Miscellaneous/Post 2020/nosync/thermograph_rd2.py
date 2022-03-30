#!/usr/bin/env python

"""
Thermography Interface

This script grabs data from the FLIR Lepton thermal camera and saves to a file
"""

import os
import datetime
import time
import numpy as np
import cv2
import argparse
from pylepton import Lepton
import RPi.GPIO as gpio

parser = argparse.ArgumentParser(description="collect and log data from the FLIR Lepton thermal camera")
parser.add_argument("--dir", type=str, default="data",
                        help="relative path to save the data")
parser.add_argument("--loop", help="repeatedly save data",
                        action="store_true")
opts = parser.parse_args()

SAVEDIR = os.path.join(os.getcwd(),opts.dir,"thermography") # path to the directory to save files
if not os.path.exists(SAVEDIR):
  print("Creating directory: {}".format(SAVEDIR))
  os.makedirs(SAVEDIR)

def save_data(data,fname):
  data_surf=data[7:50]
  data_tube=data[1:6]
  Ts=np.true_divide(np.amax(data_surf),100)-273
  Tt=np.true_divide(np.amax(data_tube),100)-273
  print(fname,",",Ts, ",",Tt)
 
if __name__ == "__main__":
  gpio.setmode(gpio.BOARD)
  gpio.setup(35, gpio.OUT)
  gpio.output(35, gpio.HIGH)

  run = True
  while run:
    try:
      with Lepton("/dev/spidev0.1") as l:
        data,_ = l.capture(retry_limit = 3)
      if l is not None:
        for line in data:
          l = len(line)
          if (l != 80):
            print("error: should be 80 columns, but we got {}".format(l))
        curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
        fname = "{}".format(curtime)
        save_data(data,fname)
        time.sleep(0.1)
        run = opts.loop
    except:
      print("an error occurred. hardware restart...")
      gpio.output(35, gpio.HIGH)
      time.sleep(0.5)
      gpio.output(35, gpio.LOW)
      print("hardware restart completed")
