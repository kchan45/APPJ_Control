
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
from scipy.interpolate import interp1d
import pickle
import sklearn

# Import core code.
import core

#import model_v1 as jet
#import EKF_v1 as observer

## load classifier
f_class=open('kmeans2.pkl','rb')
kmeans=pickle.load(f_class, encoding="latin1")

f_reg=open('linear_Trot.pkl','rb')
cl=pickle.load(f_reg, encoding="latin1")

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
#t_int=12000s
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
SAVEDIR_spec = os.path.join(os.getcwd(),runopts.dir,"{}_spectroscopy-{}".format(curtime1,runopts.tag)) # path to the directory to save thermography
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
  time.sleep(0.0200)
  #subprocess.run('echo "w,{:.2f}" > /dev/arduino_m'.format(Vn), shell=True) #firmware v14
  subprocess.run('echo "v,{:.2f}" > /dev/arduino_m'.format(Vn), shell=True) #firmware v12
  time.sleep(0.0200)
  subprocess.run('echo "y,{:.2f}" > /dev/arduino_c'.format(Yn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "x,{:.2f}" > /dev/arduino_c'.format(Xn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "d,{:.2f}" > /dev/arduino_c'.format(Dn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "p,{:.2f}" > /dev/arduino_m'.format(Pn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "f,{:.2f}" > /dev/arduino_m'.format(Fn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "q,{:.2f}" > /dev/arduino_m'.format(Qn), shell=True)
  time.sleep(0.0200)
  subprocess.run('echo "o,{:.2f}" > /dev/arduino_m'.format(On), shell=True)
  time.sleep(0.0200)
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
        mm=a ######################################## THIS NEEDS CALIBRATION #############
        ##print(data[10:80]);
        Ts = NP.amax(data[6:50,:,0]) / 100 - 273;
        #Ts=data[25,58,0]/100-273 #calibrated for the long jet
        #print(Ts_max, Ts)
        Tt = NP.amax(data[0:5,:,0]) / 100 - 273;
        #mm= NP.where( data == NP.amax(data) )
        Ts_lin=data[int(mm[0]),:,0] /100 - 273

        #yy=Ts_lin-Ts_lin[0]
        #gg=interp1d(yy,range(80))
        #sig=gg(0.6*NP.amax(yy))-mm[1]
        #print('sig',sig)
        Ts2 = (Ts_lin[int(mm[1])+2]+Ts_lin[int(mm[1])-2])/2
        Ts3 = (Ts_lin[int(mm[1])+12]+Ts_lin[int(mm[1])-12])/2
        Ts_lin_out=Ts_lin[int(mm[1])-13:int(mm[1])+13]

        for line in data:
          l = len(line)
          if (l != 80):
            print("error: should be 80 columns, but we got {}".format(l))
          elif Ts > 150:
            print("Measured temperature is too high: {}".format(Ts))

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
            data=[wv,sp_int]
            curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
            fname = "{}".format(curtime)
            save_data(SAVEDIR_spec,data,fname)
        

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
  return [Ts, Ts2, Ts3, Ts_lin_out, Tt, sig, sig2, data]

async def get_intensity_a(f,f2,runopts):
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
    T_emb=0
    P_emb=0
    Dc=0
    I_emb=0
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
          P_emb=float(line.split(',')[14])
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
    int_n=int_N2CB/max(int_N2CB)

    r=kmeans.predict(int_n.reshape(1,-1))
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

    return [O777,O844,N391,He706,sum_int,r,Tspec[0]]


async def asynchronous_measure(f,instr,runopts,max_pt):

        tasks=[asyncio.ensure_future(get_temp_a(runopts,max_pt)),
              asyncio.ensure_future(get_intensity_a(f,f_move,runopts))]

        await asyncio.wait(tasks)
        return tasks


########################################################### SET UP FOR MAIN LOOP ###############################

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

Ard_out=a[1].result()
Is=Ard_out[0]
v_rms=Ard_out[1]
x_pos=Ard_out[3]
y_pos=Ard_out[4]
d_sep=Ard_out[5]


delta=3.0
Ts_lin_old=Ts_lin
#print(Ts)
#print(P)
msg="Temperature: {:.2f} Power: {:.2f}".format(Ts,1)
print('Measurment working...')
CEM=np.zeros(np.shape(Ts_lin)) #initialize CEM measurement

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
    c.settimeout(1.)

    print('Got connection from', addr)

u_ub=[10.,20,4.,100.]
u_lb=[6.,10.,1.,100.]

############################## INITAL INPUT PARAMETERS ####################################
V=6 #initial applied voltage
#V=1.5 #initial applied power
V_list=[1.5,3.5, 3.0, 3.7, 4.5, 3.5, 2.0, 3.5]
V_list=[1.5, 2.0, 2.5, 3.5]
Dsep_list=[8.0,4.0,4.0,8.0,4.0,4.0,8.0,4.0]
Xpos_list=[-10,0,0.0,0.0,]
O=1.
Tset=40*0. #initial setpoint
Dc=100 #initial duty cycle
F=20. #initial frequency
Q=1.5 #initial flow


######################################### set position parameters ####################
t_el=0  #seconds sup. control timer
tm_el=0
Y_pos=0. #initial Y position
X_pos=0. #initial X position

#t_move=7.4 #seconds movement time
#t_move=30.0

Delta_y=-11. #mm
#Delta_y=5. #mm
#_elps=0 #seconds movement timer
t_mel=0 #PI control timer
I1=0
I2=0
Dsep=4.0
Y_dir=-1

t_dis=30. #for disturbance
t_step=1.5*60. # for step tests

t_move=45.
move=0

ind_i=0
############ initialize jet position
send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)
print('initializing jet position...')
time.sleep(5.)
############ initialize save document ################################
sv_fname = os.path.join(runopts.dir,"PI_Server_Out_{}-{}".format(curtime1,runopts.tag))
save_fl=open(sv_fname,'a+')
save_fl.write('time,Tset,Ts,Ts2,Ts3,P,Imax,Ip2p,O777,O845,N391,He706,sum_int,*U_m,q_o,D_c,x_pos,y_pos,T_emb,V,P_emb,Prms,Rdel,Is,sig,subs_type,Rot,tm_el\n')

t_move_now=time.time() ##movement_start
t_move_dis=time.time() ##disturbance_start
t_move_step=time.time() #step_start
while True:
    try:

        try:

            #data=c.recv(512).decode()
            data=c.recv(512).decode()
            data_str=data.split(',')
            data_flt=[float(i) for i in data_str]
            V=data_flt[0]
            if V<1.1:
                V=1.1
            elif V>11.:
                V=11.
           # MATT ML ONE
           #t_el=data_flt[1]
            Q=data_flt[1]
            if Q<1.0:
                Q=1.0
            elif Q>8.:
                Q=8.
           # MATT ML
            print('Optimal Reference Recieved!')
            print('Tref: {:6.2f}, t_el:{:6.2f}'.format(Tset,t_el))
        except Exception as e :
            print(e)
            print('no data yet')


#        data=c.recv(512).decode()
#        data_str=data.split(',')
#        data_flt=[float(i) for i in data_str]
#        V=data_flt[1]
#        if V<1.1:
#            V=1.1
#        elif V>5.:
#            V=5.
            # MATT ML ONE
            #t_el=data_flt[1]
#        Q=data_flt[0]
            
        print('Optimal Reference Recieved!')
        print('Tref: {:6.2f}, t_el:{:6.2f}'.format(Tset,t_el))
      
        ######################### Send inputs #########################
        print("Sending inputs...")
       # send_inputs_v_only(f,f_move,V,Y_pos,Dc,Dsep).
        send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)
        print("Inputs sent!")

        t0=time.time()
        a=ioloop.run_until_complete(asynchronous_measure(f,instr,runopts,max_pt))

        Temps=a[0].result()
        Ts_k=Temps[0]
        Ts2=Temps[1]
        Ts3=Temps[2]
        Ts_lin_k=Temps[3]
        Tt=Temps[4]
        sig_k=Temps[5]
        sig2=Temps[6]
        ## filter
        Ts=Ts*0.7+Ts_k*0.3
        Ts_lin=Ts_lin*0.7+Ts_lin_k*0.3
        
       # if sig<3: sig=sig_k 
       # sig=0.4*sig+0.6*sig_k
        sig=sig_k #no filter

        Ard_out=a[1].result()
        Is=Ard_out[0]
        v_rms=Ard_out[1]
        U_m=Ard_out[2]
        x_pos=Ard_out[3]
        y_pos=Ard_out[4]
        d_sep=Ard_out[5]
        T_emb=Ard_out[6]
        P_emb=Ard_out[7]*1.4142
        D_c=Ard_out[8]
        q_o=Ard_out[9]

        Osc_out=0
        P=0
        Ip2p=0
        Prms=0
        Rdel=0
        Imax=0

        Spec_out=0
        O777=0
        O845=0
        N391=0
        He706=0
        sum_int=0
        r=0
        Tspec=0

        Zmeas=0


        print("Temperature: {:.2f} Power: {:.2f}".format(Ts,P_emb))
        print("Inputs:{:.2f},{:.2f},{:.2f},{:.2f}".format(*U_m))
        if abs(Ts)>90:
            Ts=Ts_old
            Ts2=Ts2_old
            Ts3=Ts3_old
            Ts_lin=Ts_lin_old
            print('WARNING: Old value of Ts is used')
        else:
            Ts_old=Ts
            Ts2_old=Ts2
            Ts3_old=Ts3
            Ts_lin_old=Ts_lin
            print('WARNING: Old value of Ts is used')

         ########################### MOVE BACK AND FORTH ##############################

        print('time remaining {:6.2f}'.format(t_move-time.time()+t_move_now))
 
#        if X_pos==0: #for case I and II
#            X_dir=-1
#        if X_pos==11: #for case I and II
#            X_dir=1
#        if X_dir==1 and (time.time()-t_move_now)>=t_move: #for case I and II
#                print('Moving')
#                X_pos=X_pos+Delta_y
#                t_move_now=time.time()
#        elif X_dir==-1 and (time.time()-t_move_now)>=t_move: #for case I and II
#                print('Moving')
#                X_pos=X_pos-Delta_y
#                t_move_now=time.time()

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

   #     print(move)
   #     print((time.time()-t_move_now)-t_move)
          ####################### DISTURBANCE ########################################
  #      if (time.time()-t_move_dis)>=t_dis:
  #          print('Moving')
  #          Dsep=Dsep+2
  #          t_move_dis=time.time()

   #     if (time.time()-t_move_dis)>=t_dis:
   #         print('Moving')
   #        if Dsep==4.:
   #            Dsep=6. 
   #         else:
   #             Dsep=4.
   #         t_move_dis=time.time()

          ####################### STEP TEST ########################################
        print('time remaining {:6.2f}'.format(t_step-time.time()+t_move_step))

        if (time.time()-t_move_step)>=t_step:
        #   #Dc=Dc-15
       #      V=V_list[ind_i]
        #     X_pos=Xpos_list[ind_i]
        #     Q=Q+0.5
        #     Dsep=Dsep_list[ind_i]
        #    #O=O+0.5
             ind_i=ind_i+1
             t_move_step=time.time()


#        ######################### Send inputs #########################
#        print("Sending inputs...")
       # send_inputs_v_only(f,f_move,V,Y_pos,Dc,Dsep).
#        send_inputs_all(f,f_move,V,F,Q,Y_pos,X_pos,Dc,Dsep,O)
#        print("Inputs sent!")

        ##interpolate temperature to shift position
#        x_gen=range(26) #range of points controlled  [0-25mm]
#        x_now=NP.linspace(-13.0*2.89,12.0*2.89,26)-1+Y_pos #positions corresponding to current measurement


        #Tshift=interp1d(x_now,Ts_lin,bounds_error=False,fill_value=min(Ts_lin))(x_gen)
       # print(Ts_lin)
       # CEM=CEM+tm_el*(9.74e-14/60.0)*np.exp(np.multiply(0.6964,Tshift))
       # msg='{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}\n'.format(*CEM)
       # print(msg)

        msg='{:6.2f},{:6.2f}'.format(Ts,P_emb)
        if not runopts.auto:
            c.send(msg.encode())
            print('Measured outputs sent')

        tm_el=time.time()-t0
        print(tm_el)
      #  if tm_el<1.3:
      #      time.sleep(1.3-tm_el)
      #      tm_el=1.3

        save_fl.write('{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}\n'.format(time.time(),Tset,Ts,Ts2,Ts3,P,Imax,Ip2p,O777,O845,N391,He706,sum_int,*U_m,q_o,D_c,x_pos,y_pos,T_emb,V,P_emb,Prms,Rdel,Is,sig,sig2,0,0,tm_el))
        save_fl.flush()

        if KeyboardInterrupt==1:
            sys.exit(1)

    except Exception as e:
        if not runopts.auto:
            c.close()
        print('There was an error !')
        print(e)
        sys.exit(1)
