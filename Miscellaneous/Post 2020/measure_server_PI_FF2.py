
# Do not write bytecode to maintain clean directories
import sys
sys.dont_write_bytecode = True

# Imports required packages.
from casadi import *
import numpy as NP
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator
from scipy import linalg
from scipy import signal
from numpy import random
from scipy import io as sio
import matplotlib.pyplot as plt
import os
import datetime
import time
import cv2
import argparse
from pylepton import Lepton
import RPi.GPIO as gpio
gpio.setwarnings(False)
import subprocess
import select
import scipy.io as scio
from scipy.optimize import curve_fit
import serial
import crcmod
import visa
sys.path.append('/home/brandon/repos/python-seabreeze')
import seabreeze.spectrometers as sb
import asyncio
import usbtmc
import socket
import pickle
import sklearn
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
# Import core code.
import core

#import model_v1 as jet
#import EKF_v1 as observer

## load classifier
f_class=open('kmeans2.pkl','rb')
kmeans=pickle.load(f_class, encoding="latin1")

f_reg=open('linear_Trot.pkl','rb')
cl=pickle.load(f_reg, encoding="latin1")

f_reg=open('knn_subs.pkl','rb')
knn=pickle.load(f_reg, encoding="latin1")

crc8 = crcmod.predefined.mkCrcFun('crc-8-maxim')


##initialize oscilloscope
instr = usbtmc.Instrument(0x1ab1, 0x04ce)
instr.open()
while not (instr.timeout == 0.5 and instr.rigol_quirk == False):
    instr.timeout = 0.5
    instr.rigol_quirk = False
idg = ''
while not idg:
    try:
        idg = instr.ask("*IDN?")
    except Exception as e: # USBErrors
         print("{} in get_oscilloscope".format(e))
         time.sleep(0.4)
print("device info: {}".format(idg))
print("device timeout: {}".format(instr.timeout))
time.sleep(0.5)

## initialize spectrometer
devices = sb.list_devices()
#t_int=12000
#t_int=12000*8 ##this is what it was when training ML
t_int=12000*6
print("Available devices {}".format(devices))
spec = sb.Spectrometer(devices[0])
print("Using {}".format(devices[0]))
spec.integration_time_micros(t_int)
print("integratiopn time {} seconds.".format(t_int/1e6))
time.sleep(0.5)

class DummyFile(object):
    def write(self, x): pass

def nostdout(func):
    def wrapper(*args, **kwargs):
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        func(*args, **kwargs)
        sys.stdout = save_stdout
    return wrapper

def get_runopts():
  """g
  Gets the arguments provided to the interpreter at runtime
  """
  parser = argparse.ArgumentParser(description="runs MPC",
			  epilog="Example: python mpc_lin_test.py --quiet")
  #parser.add_argument("--quiet", help="silence the solver", action="store_true")
  parser.add_argument("--faket", help="use fake temperature data", action="store_true")
  parser.add_argument("--fakei", help="use fake intensity data", action="store_true")
  parser.add_argument("--timeout", type=float, help="timout (seconds) on oscilloscope operations",
                            default=0.4)
  parser.add_argument("--save_therm", help="save thermography photos", action="store_true")
  parser.add_argument("--save_spec", help="save OES spectra", action="store_true")
  parser.add_argument("--save_osc", help="save current waveform", action="store_true")
  parser.add_argument("--dir", type=str, default="data",help="relative path to save the data")
  parser.add_argument("--tag", type=str, default="",help="tag the saved files for easy recognition")
  parser.add_argument("--auto",help="run the code without connection to laptop", action="store_true")
  runopts = parser.parse_args()
  return runopts

##define input zero point
#U0 = NP.array([(8.0,16.0,1.2,40)], dtype=[('v','>f4'),('f','>f4'),('q','>f4'),('d','>f4')])
runopts = get_runopts()
curtime1 = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")

SAVEDIR_therm = os.path.join(os.getcwd(),runopts.dir,"{}_thermography-{}".format(curtime1,runopts.tag)) # path to the directory to save thermography files
SAVEDIR_spec = os.path.join(os.getcwd(),runopts.dir,"{}_spectroscopy-{}".format(curtime1,runopts.tag)) # path to the directory to save thermograph
SAVEDIR_osc = os.path.join(os.getcwd(),runopts.dir,"{}_oscilloscope-{}".format(curtime1,runopts.tag)) # path to the directory to save oscilloscope wavefomr current

if runopts.save_therm and not os.path.exists(SAVEDIR_therm):
  print("Creating directory: {}".format(SAVEDIR_therm))
  os.makedirs(SAVEDIR_therm)

if runopts.save_spec and not os.path.exists(SAVEDIR_spec):
  print("Creating directory: {}".format(SAVEDIR_spec))
  os.makedirs(SAVEDIR_spec)

if runopts.save_osc and not os.path.exists(SAVEDIR_osc):
  print("Creating directory: {}".format(SAVEDIR_osc))
  os.makedirs(SAVEDIR_osc)

def save_data(SAVEDIR,data,fname):
  print("saving {}.csv".format(os.path.join(SAVEDIR,fname)))
  np.savetxt(os.path.join(SAVEDIR,"{}.csv".format(fname)),data,delimiter=',',fmt='%.4e')

def send_inputs_direct(device,U):
  """
  Sends input values to the microcontroller to actuate them
  """
  Vn = U[0]
  Fn = U[1]
  Qn = U[2]
  Dn = U[3]
  Xn = U[4]
  Yn = U[5]
  input_string='echo "v,{:.2f}" > /dev/arduino && echo "f,{:.2f}" > /dev/arduino && echo "q,{:.2f}" > /dev/arduino'.format(Vn, Fn, Qn)
  #subprocess.run('echo -e "v,{:.2f}\nf,{:.2f}\nq,{:.2f}" > /dev/arduino'.format(U[:,0][0]+8, U[:,1][0]+16, U[:,2][0]+1.2), shell=True)
  device.reset_input_buffer()
  device.write("v,{:.2f}\n".format(Vn).encode('ascii'))
  time.sleep(0.200)
  device.write("f,{:.2f}\n".format(Fn).encode('ascii'))
  time.sleep(0.200)
  device.write("q,{:.2f}\n".format(Qn).encode('ascii'))
  time.sleep(0.200)
  device.write("d,{:.2f}\n".format(Dn).encode('ascii'))
  time.sleep(0.400)
  device.write("x,{:.2f}\n".format(Xn).encode('ascii'))
  time.sleep(0.200)
  device.write("y,{:.2f}\n".format(Yn).encode('ascii'))
  time.sleep(0.200)

  print("input values: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(Vn,Fn,Qn,Dn,Xn,Yn))

def send_inputs(device,U):
  """
  Sends input values to the microcontroller to actuate them
  """
  Vn = U[0]
  Fn = U[1]
  Qn = U[2]
  Dn = U[3]
  Xn = U[4]
  Yn = U[5]
  Pn = U[6]
  input_string='echo "v,{:.2f}" > /dev/arduino && echo "f,{:.2f}" > /dev/arduino && echo "q,{:.2f}" > /dev/arduino'.format(Vn, Fn, Qn)
  #subprocess.run('echo -e "v,{:.2f}\nf,{:.2f}\nq,{:.2f}" > /dev/arduino'.format(U[:,0][0]+8, U[:,1][0]+16, U[:,2][0]+1.2), shell=True)
  device.reset_input_buffer()

  subprocess.run('echo "" > /dev/arduino', shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "v,{:.2f}" > /dev/arduino'.format(Vn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "f,{:.2f}" > /dev/arduino'.format(Fn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "q,{:.2f}" > /dev/arduino'.format(Qn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "d,{:.2f}" > /dev/arduino'.format(Dn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "x,{:.2f}" > /dev/arduino'.format(Xn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "y,{:.2f}" > /dev/arduino'.format(Yn), shell=True)
  time.sleep(0.0500)

  subprocess.run('echo "p,{:.2f}" > /dev/arduino'.format(Pn), shell=True)
  time.sleep(0.0500)

  print("input values: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(Vn,Fn,Qn,Dn,Xn,Yn,Pn))

def send_inputs_v_only(device,device2,Vn,Yn,Pn,Dn):
  """
  Sends input values to the microcontroller to actuate them
  """
  device.reset_input_buffer()

  subprocess.run('echo "" > /dev/arduino_m', shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "w,{:.2f}" > /dev/arduino_m'.format(Vn), shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "y,{:.2f}" > /dev/arduino_c'.format(Yn), shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "d,{:.2f}" > /dev/arduino_c'.format(Dn), shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "p,{:.2f}" > /dev/arduino_m'.format(Pn), shell=True)
  time.sleep(0.0500)
  print("input values: V:{:.2f},Y:{:.2f}".format(Vn,Yn))

def send_inputs_all(device,device2,Vn,Fn,Qn,Yn,Xn,Pn,Dn,On):
  """
  Sends input values to the microcontroller to actuate them
  """
  device.reset_input_buffer()
  device2.reset_input_buffer()

  subprocess.run('echo "" > /dev/arduino_m', shell=True)
  time.sleep(0.0100)
  #subprocess.run('echo "v,{:.2f}" > /dev/arduino_m'.format(Vn), shell=True)  #for voltage
  subprocess.run('echo "w,{:.2f}" > /dev/arduino_m'.format(Vn), shell=True) #for power
  time.sleep(0.0100)
  subprocess.run('echo "y,{:.2f}" > /dev/arduino_c'.format(Yn), shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "x,{:.2f}" > /dev/arduino_c'.format(Xn), shell=True)
  time.sleep(0.0500)
  subprocess.run('echo "d,{:.2f}" > /dev/arduino_c'.format(Dn), shell=True)
  time.sleep(0.100)
  subprocess.run('echo "p,{:.2f}" > /dev/arduino_m'.format(Pn), shell=True)
  time.sleep(0.0100)
  subprocess.run('echo "f,{:.2f}" > /dev/arduino_m'.format(Fn), shell=True)
  time.sleep(0.0100)
  subprocess.run('echo "q,{:.2f}" > /dev/arduino_m'.format(Qn), shell=True)
  time.sleep(0.0100)
  subprocess.run('echo "o,{:.2f}" > /dev/arduino_m'.format(On), shell=True)
  time.sleep(0.0100)
  print("input values: V:{:.2f},F:{:.2f},Q:{:.2f},Y:{:.2f}".format(Vn,Fn,Qn,Yn))

def is_valid(line):
  """
  Verify that the line is complete and correct
  """
  l = line.split(',')
  crc = int(l[-1])
  data = ','.join(l[:-1])
  return crc_check(data,crc)

def crc_check(data,crc):
  crc_from_data = crc8("{}\x00".format(data).encode('ascii'))
  print("crc:{} calculated: {} data: {}".format(crc,crc_from_data,data))
  return crc == crc_from_data

def gaus(x,a,x0,sig):
    return a*NP.exp(-(x-x0)**2/(sig**2))

def get_temp_max(runopts):
  """
  Gets treatment temperature with the Lepton thermal camera
  """
  if runopts.faket:
    return 24

  run = True
  for rr in range(8):
    try:
      with Lepton("/dev/spidev0.1") as l:
        data,_ = l.capture(retry_limit = 3)
      if l is not None:
         Ts = NP.amax(data[6:50,:,0]) / 100 - 273;
         Tt = NP.amax(data[0:5,:,0]) / 100 - 273;
         mm= NP.where( data == NP.amax(data[6:50,:,0]) )
         print('max point at {},{}'.format(*mm))
         for line in data:
            l = len(line)
            if (l != 80):
                print("error: should be 80 columns, but we got {}".format(l))
            elif Ts > 150:
                print("Measured temperature is too high: {}".format(Ts))
         time.sleep(0.070)
         run = False
    except Exception as e:
      print(e)
      print("\nHardware error on the thermal camera. Lepton restarting...")
      gpio.output(35, gpio.HIGH)
      time.sleep(0.5)
      gpio.output(35, gpio.LOW)
      print("Lepton restart completed!\n\n")

  return [int(mm[0]), int(mm[1])]

############################################ ASYNC DEFS ##################################################33

async def get_temp_a(runopts,a):
  """
  Gets treatment temperature with the Lepton thermal camera
  """
  if runopts.faket:
    return 24

  run = True
  while run:
    try:
      with Lepton("/dev/spidev0.1") as l:
        data,_ = l.capture(retry_limit = 3)
      if l is not None:
        mm=a            ### This may need calibration #############
        #mm=[29,39]
        #Ts = NP.amax(data[20:40,20:40,0]) / 100 - 273;
        Ts = NP.amax(data[int(mm[0]),int(mm[1]),0]) / 100 - 273;

        ##print(data[10:80]);
        for line in data:
          l = len(line)
          if (l != 80):
            print("error: should be 80 columns, but we got {}".format(l))
          elif Ts > 150:
            print("Measured temperature is too high: {}".format(Ts))
        #Ts=data[int(mm[0]),int(mm[1]),0] / 100 - 273 #fixed_point_control
        #Ts=data[25,58,0]/100-273 #calibrated for the long jet
        #print(Ts_max, Ts)
        Tt = NP.amax(data[0:5,:,0]) / 100 - 273;
        #mm= NP.where( data == NP.amax(data[20:40,20:40,0]) )
        Ts_lin=data[int(mm[0]),:,0] /100 - 273
        yy=Ts_lin-Ts_lin[0]
        #gg=interp1d(yy,range(80))
        #sig=gg(0.6*NP.amax(yy))-mm[1]
        #print('sig',sig)
        Ts2 = (Ts_lin[int(mm[1])+2]+Ts_lin[int(mm[1])-2])/2
        Ts3 = (Ts_lin[int(mm[1])+12]+Ts_lin[int(mm[1])-12])/2
        Ts_lin_out=Ts_lin[int(mm[1])-13:int(mm[1])+13]

        Ts_2d_out=NP.array(data[int(mm[0])-13:int(mm[0])+13,int(mm[1])-13:int(mm[1])+13:,0])/100 - 273

       # sig=2*2.88/(NP.sqrt(-NP.log((Ts2-Ts3)/(Ts-Ts3))))
        sig2=2.88/(NP.sqrt(-NP.log((Ts2-Ts3)/(Ts-Ts3))))

        Ts_fit=Ts_lin_out-Ts3       #### use curve fitting for sigma calculation
        x_fit=NP.arange(-13*1.43,13*1.43,1.43)
        
        popt,pcov=curve_fit(gaus, x_fit, Ts_fit, p0=[10,0,3])
        sig=popt[2]

        if runopts.save_therm:
            curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
            fname = "{}".format(curtime)
            save_data(SAVEDIR_therm,data[:,:,0]/100.0 - 273 ,fname)

        if runopts.save_spec:
            wv=spec.wavelengths()
            sp_int=spec.intensities()
            sp_data=[wv,sp_int]
            curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
            fname = "{}".format(curtime)
            save_data(SAVEDIR_spec,sp_data,fname)

        #curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
        #fname = "{}".format(curtime)
        #Ts = NP.amax(data) / 100 - 273;
        #Ts = NP.true_divide(NP.amax(data[7:50]),100)-273;
        time.sleep(0.070)
        run = False
    except Exception as e:
      print(e)
      print("\nHardware error on the thermal camera. Lepton restarting...")
      gpio.output(35, gpio.HIGH)
      time.sleep(0.5)
      gpio.output(35, gpio.LOW)
      print("Lepton restart completed!\n\n")


  #print(Ts)
  return [Ts, Ts2, Ts3, Ts_lin_out, Tt, sig, sig2, Ts_2d_out, data]

async def get_intensity_a(f,f2,runopts): ## gettinf information from the Arduinos
  """
  Gets optical intensity from the microcontroller
  """
  if runopts.fakei:
    Is = 5
  else:
    run1 = True
    run2 = True
    v_rms=0
    Is=0
    U=[0,0,0]
    x_pos=0
    y_pos=0
    dsep=0
    P_emb=0
    I_emb=0
    T_emb=0
    Dc=0
    q_o=0
    while run1:
      try:
        f.reset_input_buffer()
        f.readline()
        line = f.readline().decode('ascii')
        if is_valid(line):
          run1 = False
          Is = int(line.split(',')[6])
          V_emb = float(line.split(',')[7])
          V = float(line.split(',')[1])
          f = float(line.split(',')[2])
          q = float(line.split(',')[3])
          Dc = float(line.split(',')[5])
          T_emb=float(line.split(',')[8])
          I_emb=float(line.split(',')[9])
          P_emb=float(line.split(',')[-2])
          q_o=float(line.split(',')[12])
        else:
          print("CRC8 failed. Invalid line!")
      #    run = False
      #    Is = 0
      #    v_rms = 0
      #    V = 0
      #    f = 0
      #    q = 0

      #  U=[V,f,q,dsep]
        U=[V,f,q,dsep]
      except:
        pass
    while run2:
      try:
        f2.reset_input_buffer()
        f2.readline()
        line = f2.readline().decode('ascii')
        if is_valid(line):
          run2 = False
          dsep=float(line.split(',')[4])
          x_pos=float(line.split(',')[10])
          y_pos=float(line.split(',')[11])
        else:
          print("CRC8 failed. Invalid line!")
      #    run = False
      #    Is = 0
      #    v_rms = 0
      #    V = 0
      #    f = 0
      #    q = 0

        U=[V,f,q,dsep]
      except:
        pass

  #print(Is)
  return [Is,v_rms,U,x_pos,y_pos,dsep,T_emb,P_emb,Dc,q_o]

def gpio_setup():
  gpio.setmode(gpio.BOARD)
  gpio.setup(35, gpio.OUT)
  gpio.output(35, gpio.HIGH)

async def get_oscilloscope_a(instr,runopts):
    try:
       # instr.write(":STOP")
        # Votlage measurement
       # instr.write(":MEAS:SOUR CHAN1")
        #Vrms=float(instr.ask("MEAS:ITEM? PVRMS"))

        instr.write(":MEAS:SOUR CHAN2")
        #Irms=float(instr.ask("MEAS:ITEM? PVRMS"))
        Imax=float(instr.ask("MEAS:VMAX?"))*1000 
        Ip2p=float(instr.ask("MEAS:VPP?"))*1000 

        rdel=float(instr.ask("MEAS:ITEM? RDEL"))
        y_raw=instr.ask(':WAV:DATA?')
        y_data=np.fromstring(y_raw[11:],dtype=float,sep=',')

        if runopts.save_osc:
            curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
            fname = "{}".format(curtime)
            save_data(SAVEDIR_osc,y_data,fname)

       #Prms=Vrms*Irms
        instr.write(":MEAS:SOUR MATH")
        P=float(instr.ask("MEAS:VAVG?"))
        Prms=float(instr.ask("MEAS:ITEM? PVRMS"))

        if P>1e3:
            print('WARNING: Measured power is too large')
            time.sleep(0.8)
            instr.write(":MEAS:SOUR MATH")
            P=float(instr.ask("MEAS:VAVG?"))

        rdel=rdel*1.e6
        fdel=0
    except Exception as e:
        print('oscilloscope error!')        
        print(e)
        P=0
        Ip2p=0
        Prms=0
        rdel=0
        Imax=0
        pass

    return [P,Ip2p,Prms,rdel,Imax]

async def get_spec_a(spec):

    wv=spec.wavelengths()
    sp_int=spec.intensities()

    int_N2CB=sp_int[63:114]-NP.mean(sp_int[-10:-1])
    int_full=sp_int[10:]-NP.mean(sp_int[-10:-1])
    int_n=int_N2CB/max(int_N2CB)

    int_full_n=int_full/max(int_full)

    r=kmeans.predict(int_n.reshape(1,-1))
    r2=knn.predict(int_full_n.reshape(1,-1))
    Tspec=cl.predict(int_n.reshape(1,-1))
    print('Trot',Tspec[0][0]*550)
    if r[0]==0:
        print('Capacitative Substrate!')
    elif r[0]==1:
        print('Conductive Substrate!')
    else:
        print('Classifier Error')
    sp_int=sp_int-NP.mean(sp_int[-20:-1])
    sum_int=NP.sum(sp_int[20:])

    ## for spec 1 - does not get OH spectrum
    O777=max(sp_int[1200:1250])
    O844=max(sp_int[1405:1455])
    N391=max(sp_int[126:146])
    He706=max(sp_int[990:1100])


    ## for spec 2 - gets OH spectrum
  #  O777=max(sp_int[1753:1777])
  #  O844=max(sp_int[1988:2001])
  #  N391=max(sp_int[563:572])
  #  He706=max(sp_int[1521:1540])
  #  OHR=max(sp_int[331:336])
  #  OHP=max(sp_int[350:355])
    #print(O777)

    return [O777,O844,N391,He706,sum_int,r,r2,Tspec[0]]

async def asynchronous_measure(f,instr,runopts,max_pt):

        tasks=[asyncio.ensure_future(get_temp_a(runopts,max_pt)),
              asyncio.ensure_future(get_intensity_a(f,f_move,runopts)),
              asyncio.ensure_future(get_oscilloscope_a(instr,runopts)),
              asyncio.ensure_future(get_spec_a(spec))]

        await asyncio.wait(tasks)
        return tasks


##################################### SET AMD TEST MEASUREMENTS ########################

Ts_old=37
Ts2_old=32
Ts3_old=25
Tslin_old=[37]*26

Pold=2
runopts = get_runopts()
gpio_setup()
f = serial.Serial('/dev/arduino_m', baudrate=38400,timeout=1)
f_move = serial.Serial('/dev/arduino_c', baudrate=38400,timeout=1)
max_pt=get_temp_max(runopts) ##get maximum point

if os.name == 'nt':
    ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(ioloop)
else:
    ioloop = asyncio.get_event_loop()

print(instr)


a=ioloop.run_until_complete(asynchronous_measure(f,instr,runopts,max_pt))

Temps=a[0].result()
Ts=Temps[0]
Ts2=Temps[1]
Ts3=Temps[2]
Ts_lin=Temps[3]
Tt=Temps[4]
sig=Temps[5]
sig2=Temps[6]
Ts_2d_out=Temps[7]
sig0=sig
Zm0=8.3

Ard_out=a[1].result()
Is=Ard_out[0]
v_rms=Ard_out[1]
x_pos=Ard_out[3]
y_pos=Ard_out[4]
d_sep=Ard_out[5]
T_emb=Ard_out[6]

Osc_out=a[2].result()
Vrms=Osc_out[0]
P=Osc_out[0]

delta=3.0
Ts_lin_old=Ts_lin
#print(Ts)
#print(P)
msg="Temperature: {:.2f} Power: {:.2f}".format(Ts,P)
print('Measurment working...')
CEM=np.zeros(np.shape(Ts_lin)) #initialize CEM measurement
#CEM2D=np.zeros(np.shape(Ts_2d_out)) #initialize 2D CEM measurement
CEM2D=np.zeros([9,9]) #initialize 2D CEM measurement
print(msg)
if not runopts.auto:
    s=socket.socket(socket.SO_REUSEADDR,1)
    host='pi3.dyn.berkeley.edu'
    port=2223
    s.bind((host,port))
    s.listen(5)
    print('\n')
    print('Server is listening on port' ,port)
    print('Waiting for connection...')

    c, addr = s.accept()
    c.settimeout(0.2)

    print('Got connection from', addr)

#u_ub=[10.,20,4.,100.]
#u_lb=[6.,10.,1.2,100.]


u_ub=[5.0,20,4.,100.]
u_lb=[1.1,10.,1.2,10.]

#u_ub=[10,20,4.,100.]
#u_lb=[6,10.,1.2,10.]


############################## INITAL INPUT PARAMETERS ####################################
V=4. #initial applied power/voltage
O=0
Dc=100. #initial duty cycle
F=20. #initial frequency
Q=1.5 #initial flow
q=1.5

x0_ff=NP.array([[0],[0],[0]])
u0_fb=V
Zm_0=0.24
Zm_k=0
############################### setpoints #####################
Tset=45. #initial setpoint
Tset_list=[40,41,40,38,43,42,45,41,40,43];
Tset_list=[40,41.5,43,44.5]
Tset_list=[44,43,45,41,40,43,44]
Tset_list=[39.2,43.6,44.01,40.3,45.]
Tset_list=[38,45,38,45,38]
Q_list=[1.5,2.0,2.5,1.2,2.8,1.8,1.5]

ind_i=1

sigset=8.5 #initial setpoint
Pset=3.  #initial setpoint

######################################### set position parameters ####################
t_el=0  #seconds sup. control timer
tm_el=0
move=0
e1=0.
e10=0.
#t_move=7.5 #seconds movement time

#t_move=30.0
Delta_y=11. #mm
#_elps=0 #seconds movement timer
t_mel=0 #PI control timer
I1=0
I2=0
Dsep=4.0
Dsep_list=[4.0, 6.2, 3.0, 5.4, 8.0, 6.8, 7.0, 4.4, 2.4, 3.0 ]
Dsep_list=[4.0, 10.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0 ,4.0, 6.0]
Y_dir=-1

t_dis=100.0
t_step=60.*2.
t_step=30.
t_move=60.*2.
int_flag=0
calib_flag=0
r_old=0
############ initialize jet position
Y_pos=0.
X_pos=0.

if runopts.auto:
    send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)
    print('initializing jet position...')
    time.sleep(5.)

############ initialize save document ################################



sv_fname = os.path.join(runopts.dir,"PI_Server_Out_{}-{}".format(curtime1,runopts.tag))
save_fl=open(sv_fname,'a+')
save_fl.write('time,Tset,Ts,Ts2,Ts3,P,Imax,Ip2p,O777,O845,N391,He706,sum_int,*U_m,q_o,D_c,x_pos,y_pos,T_emb,V,P_emb,Prms,Rdel,Is,sig,subs_type,Rot,tm_el\n')


###################################### MAIN LOOP ###########################################33

t_move_now=time.time() ##movement_start
t_move_dis=time.time() ##disturbance_start
t_move_step=time.time() #step_start
while True:
    try:
        t0=time.time()

        ########################### MOVE BACK AND FORTH ##############################

        print('time remaining {:6.2f}'.format(t_move-time.time()+t_move_now))
 
#        if X_pos==0: #for case I and II
#            X_dir=-1
#        if X_pos==-11: #for case I and II
#            X_dir=1
#        if X_dir==1 and (time.time()-t_move_now)>=t_move: #for case I and II
#                print('Moving')
#                X_pos=X_pos+Delta_y
#                t_move_now=time.time()
#        elif X_dir==-1 and (time.time()-t_move_now)>=t_move: #for case I and II
#                print('Moving')
#                X_pos=X_pos-Delta_y

        ########################### MOVE TO AND STAY ##############################        

#        if (time.time()-t_move_now)>=t_move:
#           move=1

#        if Y_dir==1 and Y_pos<=44 and move==1: #for case I and II
#                print('Moving')
#                Y_pos=Y_pos+Delta_y

#        elif Y_dir==-1 and Y_pos>=8 and move==1: #for case I and II
#                print('Moving')
#                Y_pos=Y_pos-Delta_y

#        if round(Y_pos)<=8 and move==1:
#            Y_dir=1
#            move=0
#            t_move_now=time.time()
#        if round(Y_pos)==28 and move==1:
#            move=0
#            t_move_now=time.time()
#        if round(Y_pos)>=44 and move==1: #for case I and II
#            Y_dir=-1
#            move=0
#            t_move_now=time.time()


          ####################### STEP TEST ########################################
        print('time remaining {:6.2f}'.format(t_step-time.time()+t_move_step))

      #  if (time.time()-t_move_step)>=t_step:
          #Tset=Tset+1.5
      #    Tset=Tset_list[ind_i]
       #   Q=Q_list[ind_i]
      #    ind_i=ind_i+1

      #        t_move_step=time.time()

################################# Disturbance #################################
        if (time.time()-t_move_step)>=t_step:
          #Tset=Tset+1.5
          Tset=Tset_list[ind_i]
         # Q=Q_list[ind_i]
        #  Dsep=10. 
        #  print('yaaay!')
          ind_i=ind_i+1

          t_move_step=time.time()

          ####################### RECIEVE ########################################
        try:
 
            data=c.recv(512).decode()
            data_str=data.split(',')
            data_flt=[float(i) for i in data_str]
            Tset=data_flt[0]
            #Vover=data_flt[0]
            X_pos=data_flt[1]
            Y_pos=data_flt[2]
            int_flag=data_flt[3]
            t_el=data_flt[4]

            print('Optimal Reference Recieved!')
            print('Tref: {:6.2f}, t_el:{:6.2f}'.format(Tset,t_el))
        except:
            print('no data yet')

        ######################### Send inputs #########################
        print("DSEP=",Dsep)
        print("Sending inputs...")
        #send_inputs_v_only(f,f_move,V,Y_pos,Dc,Dsep)
       # if int_flag==1:
       #     V=7.
       # V=7.        
        send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)

        print("Inputs sent!")


        if int_flag==1 and calib_flag==0:
            max_pt=get_temp_max(runopts) ##get maximum point
            calib_flag=1

        a=ioloop.run_until_complete(asynchronous_measure(f,instr,runopts,max_pt))


        Temps=a[0].result()
        Ts_k=Temps[0]
        Ts2=Temps[1]
        Ts3=Temps[2]
        Ts_lin_k=Temps[3]
        Tt=Temps[4]
        sig_k=Temps[5]
        sig2=Temps[6]
        Ts_2d_out_k=Temps[7]


       # if sig<3: sig=sig_k
       # sig=0.4*sig+0.6*sig_k
        sig=sig_k  #no filter

        Ard_out=a[1].result()
        Is=Ard_out[0]
        v_rms=Ard_out[1]
        U_m=Ard_out[2]
        x_pos=Ard_out[3]
        y_pos=Ard_out[4]
        d_sep=Ard_out[5]
        T_emb=Ard_out[6]
        P_emb=Ard_out[7]
        D_c=Ard_out[8]
        q_o=Ard_out[9]

        Osc_out=a[2].result()
        P=Osc_out[0]
        Ip2p=Osc_out[1]
        Prms=Osc_out[2]
        Rdel=Osc_out[3]
        Imax=Osc_out[4]

        Spec_out=a[3].result()
        O777=Spec_out[0]
        O845=Spec_out[1]
        N391=Spec_out[2]
        He706=Spec_out[3]
        sum_int=Spec_out[4]
        r=Spec_out[5]
        r2=Spec_out[6]
        if r2[0]>1.:
            r2[0]=1.
        Tspec=Spec_out[7]

        if Ip2p==0:
            P=P_old
            Ip2p=Ip2p_old
            Prms=Prms_old
            Rdel=Rdel_old
            Imax=Imax_old
        else:
            P_old=P
            Ip2p_old=Ip2p
            Prms_old=Prms
            Rdel_old=Rdel
            Imax_old=Imax

        Zmeas=float(U_m[0]/Ip2p/4)

        print("Temperature: {:.2f}, Sigma: {:.2f}, Power: {:.2f}".format(Ts,sig,P))
        print("Inputs:{:.2f},{:.2f},{:.2f},{:.2f}".format(*U_m))
        if abs(Ts)>60:
            Ts_k=Ts_old
            Ts2=Ts2_old
            Ts3=Ts3_old
            Ts_lin_k=Ts_lin_old
            Ts_2d_out_k=Ts_2d_old
            print('WARNING: Old value of Ts is used')
        else:
            Ts_old=Ts_k
            Ts2_old=Ts2
            Ts3_old=Ts3
            Ts_lin_old=Ts_lin_k
            Ts_2d_old=Ts_2d_out
        if abs(P)>10:
            P=Pold
            print('WARNING: Old value of Ts is used')
        else:
            Pold=P


        ## filter for temperature
        Ts=Ts*0.7+Ts_k*0.3
        Ts_lin=Ts_lin*0.7+Ts_lin_k*0.3
        Ts_2d_out=0.7*Ts_2d_out+Ts_2d_out_k*0.3

       # print(len(Ts_2d_out))
        ##interpolate temperature to shift position
        #x_gen=range(26)
        x_gen=NP.linspace(0,16,9) #range of points controlled  [0-11mm]
        y_gen=NP.linspace(0,16,9) #range of points controlled  [0-11mm]
        #x_now=NP.linspace(-.0*1.44,12.0*1.44,26)+Y_pos #positions corresponding to current measurement

        x_now=NP.linspace(-13*1.44,13*1.44,26)+y_pos-0.5 #positions corresponding to current measurement        
        y_now=NP.linspace(13*2,-13*2,26)+x_pos+0.0 #positions corresponding to current measurement

        Tshift=interp1d(x_now,Ts_lin,bounds_error=False,fill_value=min(Ts_lin))(x_gen)
        Tshift2D=interp2d(x_now,y_now,Ts_2d_out,bounds_error=False,fill_value=18,kind='cubic')(x_gen,y_gen)

        #Ts=NP.amax(Tshift2D)

        if int_flag==1:
            CEM2D=CEM2D+1.3*(9.74e-14/60.0)*np.exp(np.multiply(0.6964,Tshift2D))
            #CEM2D=CEM2D+1.3*(2.252e-15/60.0)*np.exp(np.multiply(0.7486,Tshift2D))
        ############### FEED FORWARD ####################################

        #sigma as disturbance
    #    ff_a=NP.array([[2.12,-1.287,0.333],[1, 0.0, 0.0],[0.0, 0.5, 0.0]])
    #    ff_b=NP.array([[4.0],[0.0],[0.0]])
    #    ff_c=NP.array([[0.457,-0.9448,0.977]])

       # ff_a=NP.array([[2.364,-0.88,0.3965],[2, 0.0, 0.0],[0.0, 0.5, 0.0]]) #bigger filter
       # ff_b=NP.array([[2.0],[0.0],[0.0]])
       # ff_c=NP.array([[0.6434,-0.6652,0.6835]])
        #sig0=8.
    #    xk_ff=ff_a.dot(x0_ff)+ff_b.dot(sig-sig0)
    #    sig0=sig

        #impdeance as disturbance
     #   Zm=U_m[0]/Ip2p*10

     #   ff_a=NP.array([[2.127,-1.295,0.0336],[1, 0.0, 0.0],[0.0, 0.5, 0.0]])
     #   ff_b=NP.array([[4.0],[0.0],[0.0]])
     #   ff_c=NP.array([[0.6498,-1.331,1.357]])

     #   xk_ff=ff_a.dot(x0_ff)+ff_b.dot(Zm-Zm_0)
     #   Zm_0=Zm


     # ML substrate detector as disturbance measurement
      #  ff_a=NP.array([[1.902,-1.091, 0.3526],[1, 0.0, 0.0],[0.0, 0.5, 0.0]])
        #ff_b=NP.array([[4.0],[0.0],[0.0]])
        #ff_c=NP.array([[-0.4897,1.004,-1.021]])
      #  ff_b=NP.array([[2.0],[0.0],[0.0]])
      #  ff_c=NP.array([[-0.682,1.41,-1.447]])

   #     ff_a=NP.array([[1.63,-0.6469, 0.],[1, 0.0, 0.0],[0.0, 0.0, 0.0]]) #metal no filter
   #     ff_b=NP.array([[1.0],[0.0],[0.0]])
   #     ff_c=NP.array([[0.8211,-0.7768,0]])
   #     ff_d=NP.array([-1.875])
   
  #      ff_a=NP.array([[1.683,-0.7046, 0.07168],[1, 0.0, 0.0],[0.0, 0.125, 0.0]])
  #      ff_b=NP.array([[4.0],[0.0],[0.0]])
  #      ff_c=NP.array([[-0.4199,0.8581,-3.489]])
  #      ff_d=NP.array([0])
   
        ff_a=NP.array([[2.471,-1.009, 0.5438],[2, 0.0, 0.0],[0.0, 0.5, 0.0]])
        ff_b=NP.array([[2.0],[0.0],[0.0]])
        ff_c=NP.array([[0.7591,-0.6911,0.629]])
        ff_d=NP.array([-2.899])
   
      #  ff_a=NP.array([[0.6085,-0.2471, -0.1393],[0.2589, 0.957, -0.02449],[0.01138, 0.08, 0.999]])
      #  ff_b=NP.array([[2.071],[0.3642],[0.01024]])
      #  ff_c=NP.array([[0.4584,0.5635,0.3336]])
      #  ff_d=NP.array([-4.262])

     #   ff_a=NP.array([[1.803,-1.013, 0.3795],[1, 0.0, 0.0],[0.0, 0.5, 0.0]]) #26/10/2018
     #   ff_b=NP.array([[8.0],[0.0],[0.0]])
     #   ff_c=NP.array([[-0.9294,1.639,-1.438]])
     #   ff_d=NP.array([4.784])

        xk_ff=ff_a.dot(x0_ff)+ff_b.dot(r2[0])

       #calculate the FF control action
       # yk_ff=ff_c.dot(xk_ff)+ff_d.dot(r[0])
        yk_ff=ff_c.dot(xk_ff)+ff_d.dot(r2[0])

        uk_ff=0*float(yk_ff[0])
        x0_ff=xk_ff


        #print('sigim:{}'.format(sig-sig0))
       # print('uk_ff:{}'.format(uk_ff))

        ########################## PI CONTROLS V=>T ################################
        Kp1=2.7
#        Tp1=28.8
        Tp1=15.0
        lamb1=40.0
#        lamb1=200.0

        Kc1=Tp1/(Kp1*lamb1)
        Ti1=Tp1
        e1=Tset-Ts

        Kp=2.33  ##glass
        Ki=0.186

 #       uk_fb=u0_fb+Kp*e1+(-Kp+Ki*1.3)*e10

#        u1=uk_fb#-uk_ff

#        if round(u1,2)>=u_ub[0] and Tset>=Ts:
#            I1 = I1
#            V=u_ub[0]
#        elif round(u1,2)>=u_ub[0] and Tset<Ts:
#            I1 = I1 + e1*t_mel
#            V=u_ub[0]
#        elif round(u1,2)<=u_lb[0] and Tset<=Ts:
#            I1 = I1
#            V=u_lb[0]
#        elif round(u1,2)<=u_lb[0] and Tset>Ts:
#           I1 = I1 + e1*t_mel
#           V=u_lb[0]
#        else:
#            I1=I1+e1*t_mel
#            V=round(u1,2)
         
#        e10=e1
#        u0_fb=uk_fb


        ########################## PI CONTROLS V=>P ################################
        Kp1=1.85
#        Tp1=28.8
        Tp1=1
        lamb1=2
#        lamb1=200.0

        Kc1=Tp1/(Kp1*lamb1)
        Ti1=Tp1
        e1=Pset-Prms

        u1= V+Kc1*(e1+I1/Ti1)

 #       if round(u1,2)>=u_ub[0] and Tset>=Ts:
 #           I1 = I1
 #           V=u_ub[0]
 #       elif round(u1,2)>=u_ub[0] and Tset<Ts:
 #           I1 = I1 + e1*t_mel
 #           V=u_ub[0]
 #       elif round(u1,2)<=u_lb[0] and Tset<=Ts:
 #           I1 = I1
 #           V=u_lb[0]
 #       elif round(u1,2)<=u_lb[0] and Tset>Ts:
 #          I1 = I1 + e1*t_mel
 #          V=u_lb[0]
 #       else:
 #           I1=I1+e1*t_mel
 #           V=round(u1,2)
        
        ########################## PI CONTROLS P=>T ################################
#        Kp1=4.6
        Kp1=4.5 #for metal
        Tp1=20.7
        lamb1=50.
        #lamb1=200 #for metal

        Kc1=Tp1/(Kp1*lamb1)
        Ti1=Tp1
        e1=Tset-Ts

        u1= V+Kc1*(e1+I1/Ti1)

        Kp=3.  ##glass
        Ki=0.5

        Kp=0.468 ##metal
        Ki=0.0307

        Kp=0.8454##metal
        Ki=0.02060

        Kp=6.15##glass lambda 1
        Ki=0.26

        Kp=1.785##metal lambda 2.5
        Ki=0.05307

        Kp=3.032##metal glass avg lambda 1.3
        Ki=0.1511

        Kp=1.971##metal glass avg lambda 2
        Ki=0.09821

#        Kp=1.12 ##metal glass avg lambda 3.5 22/10/18
#        Ki=0.0444

#        Kp=1.922 ##metal glass avg lambda 3.5 26/10/18
#        Ki=0.0721

        Kp=3.2##glass lambda 1.5
        Ki=0.12


#        Kp=6.15##glass lambda 1
#        Ki=0.26


#        Kp=2.976##metal lambda 1.5
#        Ki=0.08844


 #       Kp=3.07##glass lambda 2
 #       Ki=0.128


  #      Kp=2.05##glass lambda 3
  #      Ki=0.0855


#        Kp=1.49 #metal lambda 3
#        Ki=0.044

 #       Kp=0.1715 ##o2_glass
 #       Ki=0.009832

       # Kp=0.346 ##o2_glass2
       # Ki=0.02047

    
#        if r[0]-r_old==1:
#           Kp=1.3924## glass avg lambda 1
#           Ki=0.0510
#           e10=0
#           r_old=r[0]
#        elif r[0]-r_old==-1:
#           Kp=2.452## glass avg lambda 1
#           Ki=0.0932
#           e10=0
#           r_old=r[0]

#### Uncomment for power control ############################################
        uk_fb=u0_fb+Kp*e1+(-Kp+Ki*1.3)*e10

        u1=uk_fb#-uk_ff

        if round(u1,2)>=u_ub[0] and Tset>=Ts:
            I1 = I1
            V=u_ub[0]
            u0_fb=u0_fb+Kp*e1+(-Kp+0*Ki*1.3)*e10
            print('*')
        elif round(u1,2)>=u_ub[0] and Tset<Ts:
            I1 = I1 + e1*t_mel
            V=u_ub[0]
            u0_fb=uk_fb
            print('*')
        elif round(u1,2)<=u_lb[0] and Tset<=Ts:
            I1 = I1
            V=u_lb[0]
            u0_fb=u0_fb+Kp*e1+(-Kp+0*Ki*1.3)*e10
            print('*')
        elif round(u1,2)<=u_lb[0] and Tset>Ts:
           I1 = I1 + e1*t_mel
           V=u_lb[0]
           u0_fb=uk_fb
           print('*')
        else:
            I1=I1+e1*t_mel
            u0_fb=uk_fb
            V=round(u1,2)
         
        e10=e1
        #u0_fb=uk_fb
###########################################################################3
#        print(V)
        ########################## PI CONTROLS Dcycle=>T ################################

        e1=Tset-Ts
        Kp=8.305 ##o2_glass2
        Ki=0.5228

        #Kp=0.043
        #Ki=0.00802
      #  u1= Dc+Kp*e1+(-Kp+Ki*1.3)*e10

     #   if round(u1,3)>=u_ub[3] and Tset>=Ts:
     #       I1 = I1
     #       Dc=u_ub[0]
     #   elif round(u1,3)>=u_ub[3] and Tset<Ts:
     #       I1 = I1 + e1*t_mel
     #       Dc=u_ub[0]
     #   elif round(u1,3)<=u_lb[3] and Tset<=Ts:
     #       I1 = I1
     #       Dc=u_lb[0]
     #   elif round(u1,3)<=u_lb[3] and Tset>Ts:
     #      I1 = I1 + e1*t_mel
     #      Dc=u_lb[0]
     #   else:
     #       I1=I1+e1*t_mel
     #       Dc=round(u1,3)
         
     #   e10=e1

#        Kp1=0.35
#        Tp1=20.4
       # lamb1=50.
#        lamb1=200.0

#        Kc1=Tp1/(Kp1*lamb1)
#        Ti1=Tp1


#        u1=Dc+Kc1*(e1+I1/Ti1)


 #       if round(u1,3)>=u_ub[3] and Tset>=Ts:
 #           I1 = I1
 #           Dc=u_ub[3]
 #       elif round(u1,3)>=u_ub[3] and Tset<Ts:
 #           I1 = I1 + e1*t_mel
 #           Dc=u_ub[3]
 #       elif round(u1,3)<=u_lb[3] and Tset<=Ts:
 #           I1 = I1
 #           Dc=u_lb[3]
 #       elif round(u1,3)<=u_lb[3] and Tset>Ts:
 #          I1 = I1 + e1*t_mel
 #          Dc=u_lb[3]
 #       else:
 #           I1=I1+e1*t_mel
 #           Dc=round(u1,2)

     ########################## PI CONTROLS Q=>T ################################
        Kp1=-4.0
        Tp1=12
        lamb1=120.0
#        lamb1=200.0

        Kc1=Tp1/(Kp1*lamb1)
        Ti1=Tp1
        e1=Tset-Ts

        u1= q+Kc1*(e1+I1/Ti1)

 #       if round(u1,2)>=u_ub[2] and Tset>=Ts:
 #           I1 = I1
 #           q=u_ub[2]
 #       elif round(u1,2)>=u_ub[2] and Tset<Ts:
 #          I1 = I1 + e1*t_mel
 #          q=u_ub[2]
 #       elif round(u1,2)<=u_lb[2] and Tset<=Ts:
 #           I1 = I1
 #           q=u_lb[2]
 #       elif round(u1,2)<=u_lb[2] and Tset>Ts:
 #          I1 = I1 + e1*t_mel
 #          q=u_lb[2]
 #       else:
 #           I1=I1+e1*t_mel
 #           q=round(u1,2)

#        print(V)
        ########################### PI CONTROLS Q=>SIGMA ###########################3
        Kp2=4.943/9.164
        Tp2=1/9.164
        lamb2=1

        Kc2=Tp2/(Kp2*lamb2)
        Ti2=Tp2
        e2=sigset-sig

        u2=q+Kc2*(e2+I2/Ti2)


    #    if round(u2,2)>=u_ub[2] and sigset>=sig:
    #        I2 = I2
    #        q=u_ub[2]
    #    elif round(u2,2)>=u_ub[2] and sigset<sig:
    #        I2 = I2 + e1*t_mel
    #        q=u_ub[2]
    #    elif round(u2,2)<=u_lb[2] and sigset<=sig:
    #        I2 = I2
    #        q=u_lb[2]
    #    elif round(u2,2)<=u_lb[2] and sigset>sig:
    #       I2 = I2 + e2*t_mel
    #       q=u_lb[2]
    #    else:
    #        I2=I2+e2*t_mel
    #        q=round(u2,2)



###### DISTURBANCE
#        if (time.time()-t_move_dis)>=t_dis:
#            print('Moving')
#            Dsep=Dsep+2
#            t_move_dis=time.time()

#        if (time.time()-t_move_dis)>=t_dis:
#            print('Moving')
#            if Dsep==4.:
#               Dsep=6.
#            else:
#                Dsep=4.
#            t_move_dis=time.time()


        ######################### Send inputs #########################
#        print("DSEP=",Dsep)
#        print("Sending inputs...")
        #send_inputs_v_only(f,f_move,V,Y_pos,Dc,Dsep)
       # if int_flag==1:
       #     V=7.
       # V=7.        
#        send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)

#d        print("Inputs sent!")
        

        #print(*CEM2D.flatten().shape)
        #CEM=CEM+tm_el*(9.74e-14/60.0)*np.exp(np.multiply(0.6964,Tshift))


#        msg='{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f} \n'.format(*CEM)
        
        msg_form='{:6.2f},'
        msg_str=msg_form*(1+len(Tshift2D.flatten()))
        msg_str=msg_str[:-1]+'\n'
        msg=msg_str.format(*CEM2D .flatten(),Ts)

        
        msg2='{:6.2f}\n'.format(Ts)
        #print(msg2)

        if not runopts.auto:
            c.send(msg.encode())
            time.sleep(0.05)
            print('Measured outputs sent')


        tm_el=time.time()-t0

        if tm_el<1.3:
            time.sleep(1.3-tm_el)
            tm_el=1.3

        save_fl.write('{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}\n'.format(time.time(),Tset,Ts,Ts2,Ts3,P,Imax,Ip2p,O777,O845,N391,He706,sum_int,*U_m,q_o,D_c,x_pos,y_pos,T_emb,V,P_emb,Prms,Rdel,Is,sig,sig2,r[0],r2[0],Tspec[0]*550,uk_ff,tm_el))
        save_fl.flush()

        if KeyboardInterrupt==1:
            sys.exit(1)

    except Exception as e:
        if not runopts.auto:
            c.close()
        print('There was an error !')
        print(e)
        sys.exit(1)
