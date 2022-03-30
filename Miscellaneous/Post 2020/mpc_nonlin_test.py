# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:31:09 2017

@author: Dogan
"""
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
import serial
import crcmod
import visa
sys.path.append('/home/brandon/repos/python-seabreeze')
import seabreeze.spectrometers as sb
import asyncio
import usbtmc

# Import core code.
import core

import model_v1 as jet
import EKF_v1 as observer


crc8 = crcmod.predefined.mkCrcFun('crc-8-maxim')

##initialize oscilloscope
instr = usbtmc.Instrument(0x1ab1, 0x04ce)
instr.open()
while not (instr.timeout == 1 and instr.rigol_quirk == False):
    instr.timeout = 1
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
t_int=12000*4
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
  """
  Gets the arguments provided to the interpreter at runtime
  """
  parser = argparse.ArgumentParser(description="runs MPC",
			  epilog="Example: python mpc_lin_test.py --quiet")
  #parser.add_argument("--quiet", help="silence the solver", action="store_true")
  parser.add_argument("--faket", help="use fake temperature data", action="store_true")
  parser.add_argument("--fakei", help="use fake intensity data", action="store_true")
  parser.add_argument("--timeout", type=float, help="timout (seconds) on oscilloscope operations",
                            default=0.4)
  runopts = parser.parse_args()
  return runopts

##define input zero point
#U0 = NP.array([(8.0,16.0,1.2,40)], dtype=[('v','>f4'),('f','>f4'),('q','>f4'),('d','>f4')])

def send_inputs(device,U):
  """
  Sends input values to the microcontroller to actuate them
  """
  Vn = U[0]
  Fn = U[1]
  Qn = U[2]
  Dn = U[3]
  input_string='echo "v,{:.2f}" > /dev/arduino && echo "f,{:.2f}" > /dev/arduino && echo "q,{:.2f}" > /dev/arduino'.format(Vn, Fn, Qn)
  #subprocess.run('echo -e "v,{:.2f}\nf,{:.2f}\nq,{:.2f}" > /dev/arduino'.format(U[:,0][0]+8, U[:,1][0]+16, U[:,2][0]+1.2), shell=True)
  device.reset_input_buffer()
  #device.write("v,{:.2f}\n".format(Vn).encode('ascii'))
  subprocess.run('echo "" > /dev/arduino', shell=True)
  time.sleep(0.200)
  subprocess.run('echo "v,{:.2f}" > /dev/arduino'.format(Vn), shell=True)
  time.sleep(0.200)
  #device.write("f,{:.2f}\n".format(Fn).encode('ascii'))
  subprocess.run('echo "f,{:.2f}" > /dev/arduino'.format(Fn), shell=True)
  time.sleep(0.200)
  #device.write("q,{:.2f}\n".format(Qn).encode('ascii'))
  subprocess.run('echo "q,{:.2f}" > /dev/arduino'.format(Qn), shell=True)
  #subprocess.call(input_string,  shell=True)
  #print("input: {}".format(input_string))
  time.sleep(0.200)
  subprocess.run('echo "d,{:.2f}" > /dev/arduino'.format(Dn), shell=True)
  print("input values: {:.2f},{:.2f},{:.2f},{:.2f}".format(Vn,Fn,Qn,Dn))

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


############################################ ASYNC DEFS ##################################################33

async def get_temp_a(runopts):
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
        ##print(data[10:80]);
        Ts = NP.amax(data[6:60,:,0]) / 100 - 273;
        Tt = NP.amax(data[0:5,:,0]) / 100 - 273;
        mm= NP.where( data == NP.amax(data) )

        Ts_lin=data[int(mm[0]),:,0] /100 - 273
        Ts2 = (Ts_lin[int(mm[1])+2]+Ts_lin[int(mm[1])-2])/2
        Ts3 = (Ts_lin[int(mm[1])+6]+Ts_lin[int(mm[1])-6])/2
        for line in data:
          l = len(line)
          if (l != 80):
            print("error: should be 80 columns, but we got {}".format(l))
          elif Ts > 150:
            print("Measured temperature is too high: {}".format(Ts))
        #curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")
        #fname = "{}".format(curtime)
        #Ts = NP.amax(data) / 100 - 273;
        #Ts = NP.true_divide(NP.amax(data[7:50]),100)-273;
        time.sleep(0.050)
        run = False
    except:
      print("\nHardware error on the thermal camera. Lepton restarting...")
      gpio.output(35, gpio.HIGH)
      time.sleep(0.5)
      gpio.output(35, gpio.LOW)
      print("Lepton restart completed!\n\n")
  
  #print(Ts)
  return [Ts, Ts2, Ts3, Ts_lin, Tt, data]

async def get_intensity_a(f,runopts):
  """
  Gets optical intensity from the microcontroller
  """
  if runopts.fakei:
    Is = 5
  else:
    run = True
    v_rms=0
    Is=0
    U=[0,0,0]
    while run:
      try:
        f.reset_input_buffer()
        f.readline()
        line = f.readline().decode('ascii')
        if is_valid(line):
          run = False
          Is = int(line.split(',')[6])
          v_rms = float(line.split(',')[7])
          V = float(line.split(',')[1])
          f = float(line.split(',')[2])
          q = float(line.split(',')[3])
          dsep=float(line.split(',')[4])
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
  return [Is,v_rms,U]

def gpio_setup():
  gpio.setmode(gpio.BOARD)
  gpio.setup(35, gpio.OUT)
  gpio.output(35, gpio.HIGH)

async def get_oscilloscope_a(instr):

    #instr.write(":STOP")
    # Votlage measurement
   
    instr.write(":MEAS:SOUR MATH")
    P=float(instr.ask("MEAS:VAVG?"))

    if P>1e3:
        print('WARNING: Measured power is too large')
        time.sleep(0.8)
        instr.write(":MEAS:SOUR MATH")
        P=float(instr.ask("MEAS:VAVG?"))

   # instr.write(":RUN")
    time.sleep(0.4)
    
    #print(P)

    return [P]

async def get_spec_a(spec):

    wv=spec.wavelengths()
    sp_int=spec.intensities()
    O777=max(sp_int[1200:1250])
    
    #print(O777)
    return O777

def syncronous_measure(f,instr,runopts):
    Ts=get_temp(runopts)
    get_intensity(f,runopts)
    get_oscilloscope(instr)
    get_spec(spec)
    return Ts


async def asynchronous_measure(f,instr,runopts):

        tasks=[asyncio.ensure_future(get_temp_a(runopts)),
              asyncio.ensure_future(get_intensity_a(f,runopts)),
              asyncio.ensure_future(get_oscilloscope_a(instr))]
        
        await asyncio.wait(tasks)
        return tasks

############################################ INITIALIZE ###########################################

save_file=open('control_dat','a+')
### IMPORT MAT FILE###
mat=sio.loadmat('intermed2.mat')
prevopt=mat['optimal']
save_file=open('control_dat','a+')

Qx=mat['Qx']
Qz=mat['Qz']
Ru=mat['Ru']
Qt=mat['Qt']

Qk=mat['Qk']
Rk=mat['Rk']

N=(mat['N'])
Pk_1=mat['Pk']
delta=3.0

######################################## MODEL ########################
#######################################################################

[x, z, u, d, v, xdot, gn, I] = jet.model(delta)
nx=int(x.dim()[0])
nz=int(z.dim()[0])
nu=int(u.dim()[0])
nv=int(v.dim()[0])

# test integrator
#x_0=[313.0/300.0, 313.0/300.0, 313.0/300.0, 0.0, 0.0, 0.0, (313.0+20.0)/300.0,  (295.0+20.0)/300.0]
x_0=[313.0/300.0, 0.0, (313.0+20.0)/300.0,  (295.0+20.0)/300.0]

Ik = I(x0=x_0,p=[4,15,2,4],z0=[3,3,3,3]);

xk = Ik['xf'].full().flatten().tolist()
zk = Ik['zf'].full().flatten().tolist()
uk = [4.0,15.0,2.0]
dis=[0, 0]
######################## MPC ######################################
###################################################################


#uss=[4,15,2]
dss=4

N=6
u_ub = [5.0,20.0,5.0]
u_lb = [3.1,10.1,1.5]

du_ub = [0.5,1,0.5]
du_lb = [-0.8,-1,-0.8]

#x_lb = [-inf, -inf, -inf, -inf, -inf]
x_lb = [0, 0, 0, 0, 0]
x_ub = [inf, inf, inf, inf, 10]

#z_lb = [-inf, -inf, -inf, -inf]
z_lb = [0, 0, 0, 0]
z_ub = [inf , inf, inf, inf]

xset=[(60.0+273)/300.0, 1.0, (41.0+273)/300.0, (30.0+273)/300.0]
zset=[5.0, 4.5, 17.0, (60.0+273)/300.0]
#zss=[3,3,3,3]

#zss = core.rootFinder(alg, nz, args=[uss,dss])

#xss = core.rootFinder(diffs, nx, args=[zss,uss,dss], x0=x_0[0:nx-1])
### Build instance of MPC problem
# start with an empty NLP
q = []
q0 = []
lbq = []
ubq = []
J = 0
g = []
lbg = []
ubg = []
# "lift" initial conditions


# formulate the NLP
X0 = MX.sym('X0', nx)
q  += [X0]
qz = []
lbq += xk
ubq += xk
Zk = zk
q0 += xk
Xk = X0
U0 = uk

for k in range(N):
    # new NLP variable for the control
    Uk = MX.sym('U_' + str(k), nu)
    q   += [Uk]
    lbq += [u_lb[i] for i in range(nu)]
    ubq += [u_ub[i] for i in range(nu)]
    q0  += uk
    
 
    # next step dynamics 
    Fk = I(x0=Xk ,p=vertcat(Uk,4),z0=Zk)
    Xk_end = Fk['xf']
    Zk_end = Fk['zf']

    #add to stage cost
    J = J + 10*(Xk[2]-xset[2])**2.0 + 0.001*(Zk[1]-zset[1])**2 #+ 0*core.mtimes((Uk-uss).T,Qr,(Uk-uss))
    #J=J+Fk['qf'] 
   # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), nx)
    q += [Xk]
    lbq += [x_lb[i] for i in range(nx)]     
    ubq += [x_ub[i] for i in range(nx)]     
    q0 += xk
    
    Zk = MX.sym('Z_' + str(k+1), nz)
    q += [Zk]
    lbq += [z_lb[i] for i in range(nz)]     
    ubq += [z_ub[i] for i in range(nz)]     
    q0 += zk
    
    # Add equality constraint
#    g   += [Xk_end-Xk]
    g   += [Xk_end-Xk]
    lbg += [0]*nx
    ubg += [0]*nx
    
    g   += [Zk_end-Zk]
    lbg += [0]*nz
    ubg += [0]*nz
    
    g   += [Uk-U0]
    lbg += [du_lb[i] for i in range(nu)]
    ubg += [du_ub[i] for i in range(nu)]
    U0=Uk
    
#terminal cost
J = J + 0*(Xk[2]-1.04)**2 

MySolver = "ipopt"
#MySolver = "sqpmethod"

sol_opts={}

if MySolver == "sqpmethod":
  sol_opts["qpsol"] = "qpoases"
  #sol_opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == 'ipopt':
  #sol_opts={'ipopt.hessian_approximation':'limited-memory','ipopt.fixed_variable_treatment':'relax_bounds' ,'ipopt.jacobian_approximation':'finite-difference-values'}
  sol_opts={'ipopt.hessian_approximation':'limited-memory','ipopt.fixed_variable_treatment':'relax_bounds' ,'ipopt.tol':1e-4, 'ipopt.max_cpu_time':35}

 
prob = {'f':J, 'x':vertcat(*q), 'g': vertcat(*g)}
solver = nlpsol('solver', MySolver, prob, sol_opts)

sol = solver(x0=q0, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)
optimal=sol['x'].full().flatten()
###################################################  MAIN LOOP #####################################3


if __name__ == "__main__":

    runopts = get_runopts() 
    gpio_setup()
    f = serial.Serial('/dev/arduino', baudrate=38400,timeout=1)

    delay=3
    counter=0
    Ts_old=40

    if os.name == 'nt':
            ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
            asyncio.set_event_loop(ioloop)
    else:
            ioloop = asyncio.get_event_loop()
        
    
    a=ioloop.run_until_complete(asynchronous_measure(f,instr,runopts))
           # Ts=tasks[0].result()
           # print('async:',t1-t0) 
        
    ##### UNPACKING TASK OUTS ########
    Temps=a[0].result()
    Ts=Temps[0]
    Ts2=Temps[1]
    Ts3=Temps[2]
    Tt=Temps[4]
 
    Ard_out=a[1].result()   
    Is=Ard_out[0]
    v_rms=Ard_out[1] 
    U=Ard_out[2]
    Osc_out=a[2].result()
    Vrms=Osc_out[0]
    P=Osc_out[0]

    while True:
        start_time = time.time()
        #Ts=syncronous_measure(f,instr,runopts)
        #t1=time.time()
        #print('sync:',t1-t0)       
        #print(Ts)

        if os.name == 'nt':
            ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
            asyncio.set_event_loop(ioloop)
        else:
            ioloop = asyncio.get_event_loop()
        
        if counter==20:
            uk=[4.0,15.0,2.0]
            print('sending inputs..')
            send_inputs(f,[8.0, 15.0, 2.0, 60])
            print('inputs sent!')
############################################ MEASUREMENT ###########################################
        a=ioloop.run_until_complete(asynchronous_measure(f,instr,runopts))
           # Ts=tasks[0].result()
           # print('async:',t1-t0) 
        
          ##### UNPACKING TASK OUTS ########
        Temps=a[0].result()
        Ts=Temps[0]
        Ts2=Temps[1]
        Ts3=Temps[2]
        Tt=Temps[4]

        Ard_out=a[1].result()   
        Is=Ard_out[0]
        v_rms=Ard_out[1]
        U=Ard_out[2]
        Osc_out=a[2].result()
        Vrms=Osc_out[0]
        P=Osc_out[0]

                     
        print("Temperature: {:.2f} Power: {:.2f}".format(Ts,P))
        if abs(Ts)>100:
            Ts=Ts_old
            print('WARNING: Old value of Ts is used')
        else:
            Ts_old=Ts

        if Is<0:
            Is=Is_old
            print('WARNING: Old value of Is is used')
        else:
            Is_old=Is


################################### OBSERVER ###################################################

        Ik = I(x0=xk,p=vertcat(uk,4),z0=[1,1,1,1]);
        #xk = optimal[(nx+nu):(nx+nu+nx)]
        xk_pred=Ik['xf'].full().flatten()
        zk_pred=Ik['zf'].full().flatten()

        y_hat_k=[-(xk_pred[2]+dis[0]-(Ts+273)/300), -(xk[3]-(Ts3+273)/300), -(zk_pred[1]+dis[1]-P)]
  

        [xk,dis,zk,Pk]=observer.EKF(x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xk_pred,dis,zk_pred,uk)
            

        Pk_1=Pk
            
        #xk[2]=xk[2]+dis[0]
    
        print("Est. Temperature: {:.2f} Est. Power: {:.2f}".format((xk[2]+dis[0])*300-273,zk[1]+dis[1]))

      
####################################### SAVE ##############################################################################
        save_file.write("{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}\n".format(time.time(),Ts,P,*U,counter,(xset[2]*300.0-273.0),zset[1],(xk[2]+dis[0])*300.0-273.0,zk[1]+dis[1],dis[0]*300, dis[1]))  ##X is never referenced!
                  #print()
        save_file.flush()

        end_time = time.time()
        time_el = end_time - start_time

        if time_el < delta:
            time.sleep(delta - time_el)
                ## increment the loop counter 
        counter=counter+1
