# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:49:52 2017

@author: Dogan
"""

import sys
sys.path.append("C:\\Users\\Dogan\\Documents\\Casadi_Python")
#sys.path.append("C:\\Users\\Dogan\\Dropbox\\COMSOL\\Flow_model")
#sys.path.append("C:\\Users\\Dogan\\Documents\\casadi_36")
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

# Import core code.
import core

def model(delta):
    nx = 4 #states
    nu = 3 #inputs
    nv = 2 #outputdis
    nd = 1 #disturbance
    nz = 4 #algebraic states
    
    # declare casadi variable graph
    x = SX.sym('x',nx)
    z = SX.sym('z',nz)
    u = SX.sym('u',nu)
    d = SX.sym('d',nd)
    v = SX.sym('v',nv)

    
    # Helium
    propHe={'cp':5.1931E+3, #J/kgK
            'miu':2.0484E-5, #kg/m.s
             'k':0.15398, #%W/mK
             'rho':0.15608, #kg/m^3
             'Mw':4.948e-3}
             
             
    # Air
    propAir={'cp':1.00e3, #J/kgK
            'miu':1.868e-5, #kg/m.s
             'k':0.026, #%W/mK
             'rho':1.165, #kg/m^3
             'Mw':28.966e-3}
             
    #Surface
    propSurf={'rho':2.8e3, 
          'cp':795.00, 
          'k':1.43, 
          'd':0.20e-3} #surface thickness
          
    #system dimensions
    dim={'r':1.5e-3};
    dim['vol']=3.1416*1e-2*dim['r']**2 #m^3 volume of plasma chamber 
    dim['Ac']=3.1416*dim['r']**2 #m^2 flow crossectional area
    
    ######################  VARIABLES #########################3
    
    #inputs and disturbance
     
    Va=u[0]*1.00e3
    f=u[1]*1.00e3
    q=u[2]
    dsep = d[0]*1.00e-3 #m DISTURBANCE VALUE
    
    eta=0.4+0.07*dsep/4.00e-3; #FITTED
    
    #algebraic states
    ip = z[0]*1.00e-3 #current in amps
    P =  z[1]; #power in W
    Rp = z[2]*1.00e5; #plasma resistivity in Ohm
    Tin = z[3]*300.00; #peak gas temeprature in K 
    
    #differential states
    T1  = x[0]*300.00
    #T2  = x[1]*300.00
    #T3  = x[2]*300.00
    
    w1  = x[1]
    #w2  = x[4]
    #w3  = x[5]
    
    Ts_max = x[2]*300.00 #surface temeprature in K
    Ts_2   = x[3]*300.00 # outside circle surface temeprature in K
    
    
    ## Derived algebraic expressions heat transfer
    Tinf = 293.00 # K ambient temperature
    R    = 8.314 # ideal gas constant 
    Patm = 101.3e3 #Pa
    win  = 1.00 #inlet He fraction
    e1   = 0.90 #distribution coefficient
    eta  = 0.4+0.07*dsep/4.00e-3 #power deposition efficiency
    H_I  = 1.6
    
    rhoin=(Patm/(R))*(propHe['Mw'])/Tinf
    vin=q*(1.0/60.0)*0.001*(Tinf/273.0)/dim['Ac']
    
    Pow=(eta*P)/(dim['Ac']*1.00e-2) #volumetric power
    
    ## Derived algebraic expressions circuit
    n_T=1.0
    Cp=0.94003e-11
    b=20.7911e5
    om=2.0*3.1416*f
    C0=8.072e-12
    eps0=8.85e-12 #vacuum permittivity
    k_diel=4.50 #relative premittiviy of quartz
    Adiel=24.00e-3*60e-3 #area of the surface dielectric
    Cs=k_diel*eps0*Adiel/2.00e-4 #surface capacitance;
    
    #algebraic expressions
    Tincalc= Tinf+(dim['r']*Pow/(H_I*sqrt(vin)))*((1-e1)+(e1-(1-e1))*exp(-H_I*1.00e-2*sqrt(vin)/(rhoin*vin*propHe['cp']*dim['r']))-e1*exp(-(2.0*H_I*1.0e-2)*sqrt(vin)/(rhoin*vin*propHe['cp']*dim['r'])));
    Rpcalc=0.9*(b)*((Tin/340.0))
    #Rpcalc=b
    Pcalc=(Cp**2.00*Cs**2.00*Rp*ip**2.00)/(2.00*(C0**2.00*Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + C0**2.00*Cp**2.00 + 2.00*C0**2.00*Cp*Cs + C0**2.00*Cs**2.00 + 2.00*C0*Cp**2.00*Cs + 2.00*C0*Cp*Cs**2.00 + Cp**2.00*Cs**2.00));
    Vacalc=(ip*((Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + Cp**2.00 + 2.00*Cp*Cs + Cs**2.00)/(C0**2.00*Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + C0**2.00*Cp**2.00 + 2.00*C0**2.00*Cp*Cs + C0**2.00*Cs**2.00 + 2.00*C0*Cp**2.00*Cs + 2.00*C0*Cp*Cs**2.00 + Cp**2.00*Cs**2.00))**(1.00/2.00))/om;
    
    gn1 = (Va-Vacalc)/1000.00
    gn2 = (P-Pcalc)
    gn3 = (Rp-Rpcalc)/1.00e5
    gn4 = (Tin-Tincalc)/300.00
    
    # Heat and mass transfer derived experssions
    U_h=1.83*vin**(0.5) #heat transfer in gas
    K=0.017*vin**(0.5) #mass transfer in gas 
    h=50.0*vin**0.8 #heat transfer between gas and surface
    
    wAinf=0.0
    n=1.0
    
    cp1=w1*propHe['cp']+(1.0-w1)*propAir['cp'] #specific heat 
    rho1=(Patm/(R))*(propHe['Mw']*w1+propAir['Mw']*(1.0-w1))/T1 #denisty 
    
    #cp2=w2*propHe['cp']+(1.0-w2)*propAir['cp'] #specific heat 
    #rho2=(Patm/(R))*(propHe['Mw']*w2+propAir['Mw']*(1.0-w2))/T2 #denisty 
    
    #cp3=w3*propHe['cp']+(1.0-w3)*propAir['cp'] #specific heat 
    #rho3=(Patm/(R))*(propHe['Mw']*w3+propAir['Mw']*(1.0-w3))/T3 #denisty 
    
    ## differential expressions
    
    #temeprature
    dT1dt=(1.00/(rho1*cp1*(dsep/n)*dim['Ac']))*(rhoin*vin*cp1*dim['Ac']*(Tin-T1)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T1-Tinf));
    #dT2dt=(1.00/(rho2*cp2*(dsep/n)*dim['Ac']))*(rhoin*vin*cp2*dim['Ac']*(T1-T2)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T2-Tinf));
    #dT3dt=(1.00/(rho3*cp3*(dsep/n)*dim['Ac']))*(rhoin*vin*cp3*dim['Ac']*(T2-T3)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T3-Tinf));
    
    #mass fraction
    dw1dt=(1.00/(rho1*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(win-w1)-K*(dsep/n)*rho1*(2.00*3.1416*dim['r'])*(w1-wAinf));
    #dw2dt=(1.00/(rho2*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(w1-w2)-K*(dsep/n)*rho2*(2.00*3.1416*dim['r'])*(w2-wAinf));
    #dw3dt=(1.00/(rho3*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(w2-w3)-K*(dsep/n)*rho3*(2.00*3.1416*dim['r'])*(w3-wAinf));
    
    #substrate
    dTs_maxdt=(dim['Ac']*h*(T1-Ts_max)-2.0*3.1416*propSurf['d']*propSurf['k']*(Ts_max-Ts_2))/(propSurf['rho']*propSurf['cp']*dim['Ac']*propSurf['d']);
    dTs2_dt=((2.0*3.1416*propSurf['d']*propSurf['k']*(Ts_max-Ts_2)-(4.0/vin)*(4.00e-3/dsep)*3.1416*propSurf['d']*propSurf['k']*(Ts_2-Tinf))/(propSurf['rho']*propSurf['cp']*dim['Ac']*propSurf['d']*100.0));
    #dCEMT = ((9.76e-14/60)*exp(0.6964*(Ts_max-273.0)))/10.0
    dCEMT = ((9.74e-14/60)*exp(0.6964*(Ts_max-273.0)))/10.0
    
    # collate expressions and create evaluator function f and integrator I
    #xdot = vertcat(dT1dt/300.0,dT2dt/300.0,dT3dt/300.0,dw1dt,dw2dt,dw3dt,dTs_maxdt/300.0,dTs2_dt/300.0)
    xdot_ss = vertcat(dT1dt/300.0,dw1dt,dTs_maxdt/300.0,dTs2_dt/300.0)
    xdot = vertcat(dT1dt/300.0,dw1dt,dTs_maxdt/300.0,dTs2_dt/300.0)
    gn   = vertcat (gn1,gn2,gn3,gn4)
    jet_dae = {'x':x, 'p':vertcat(u,d), 'z':z, 'alg':gn, 'ode':xdot}
    opts = {'tf':delta}
    
    I = integrator('I', 'idas', jet_dae, opts)
    
    return [x, z, u, d, v, xdot, gn, I]
    
def model_w_dose():
    nx = 6 #states
    nu = 3 #inputs
    nv = 2 #outputdis
    nd = 1 #disturbance
    nz = 4 #algebraic states
    
    # declare casadi variable graph
    x = SX.sym('x',nx)
    z = SX.sym('z',nz)
    u = SX.sym('u',nu)
    d = SX.sym('d',nd)
    v = SX.sym('v',nv)

    
    # Helium
    propHe={'cp':5.1931E+3, #J/kgK
            'miu':2.0484E-5, #kg/m.s
             'k':0.15398, #%W/mK
             'rho':0.15608, #kg/m^3
             'Mw':4.948e-3}
             
             
    # Air
    propAir={'cp':1.00e3, #J/kgK
            'miu':1.868e-5, #kg/m.s
             'k':0.026, #%W/mK
             'rho':1.165, #kg/m^3
             'Mw':28.966e-3}
             
    #Surface
    propSurf={'rho':2.8e3, 
          'cp':795.00, 
          'k':1.43, 
          'd':0.20e-3} #surface thickness
          
    #system dimensions
    dim={'r':1.5e-3};
    dim['vol']=3.1416*1e-2*dim['r']**2 #m^3 volume of plasma chamber 
    dim['Ac']=3.1416*dim['r']**2 #m^2 flow crossectional area
    
    ######################  VARIABLES #########################3
    
    #inputs and disturbance
     
    Va=u[0]*1.00e3
    f=u[1]*1.00e3
    q=u[2]
    dsep = d[0]*1.00e-3 #m DISTURBANCE VALUE
    
    eta=0.4+0.07*dsep/4.00e-3; #FITTED
    
    #algebraic states
    ip = z[0]*1.00e-3 #current in amps
    P =  z[1]; #power in W
    Rp = z[2]*1.00e5; #plasma resistivity in Ohm
    Tin = z[3]*300.00; #peak gas temeprature in K 
    
    #differential states
    T1  = x[0]*300.00
    #T2  = x[1]*300.00
    #T3  = x[2]*300.00
    
    w1  = x[1]
    #w2  = x[4]
    #w3  = x[5]
    
    Ts_max = x[2]*300.00 #surface temeprature in K
    Ts_2   = x[3]*300.00 # outside circle surface temeprature in K
    
    
    ## Derived algebraic expressions heat transfer
    Tinf = 293.00 # K ambient temperature
    R    = 8.314 # ideal gas constant 
    Patm = 101.3e3 #Pa
    win  = 1.00 #inlet He fraction
    e1   = 0.90 #distribution coefficient
    eta  = 0.4+0.07*dsep/4.00e-3 #power deposition efficiency
    H_I  = 1.6
    
    rhoin=(Patm/(R))*(propHe['Mw'])/Tinf
    vin=q*(1.0/60.0)*0.001*(Tinf/273.0)/dim['Ac']
    
    Pow=(eta*P)/(dim['Ac']*1.00e-2) #volumetric power
    
    ## Derived algebraic expressions circuit
    n_T=1.0
    Cp=0.94003e-11
    b=20.7911e5
    om=2.0*3.1416*f
    C0=8.072e-12
    eps0=8.85e-12 #vacuum permittivity
    k_diel=4.50 #relative premittiviy of quartz
    Adiel=24.00e-3*60e-3 #area of the surface dielectric
    Cs=k_diel*eps0*Adiel/2.00e-4 #surface capacitance;
    
    #algebraic expressions
    Tincalc= Tinf+(dim['r']*Pow/(H_I*sqrt(vin)))*((1-e1)+(e1-(1-e1))*exp(-H_I*1.00e-2*sqrt(vin)/(rhoin*vin*propHe['cp']*dim['r']))-e1*exp(-(2.0*H_I*1.0e-2)*sqrt(vin)/(rhoin*vin*propHe['cp']*dim['r'])));
    Rpcalc=0.9*(b)*((Tin/340.0))
    #Rpcalc=b
    Pcalc=(Cp**2.00*Cs**2.00*Rp*ip**2.00)/(2.00*(C0**2.00*Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + C0**2.00*Cp**2.00 + 2.00*C0**2.00*Cp*Cs + C0**2.00*Cs**2.00 + 2.00*C0*Cp**2.00*Cs + 2.00*C0*Cp*Cs**2.00 + Cp**2.00*Cs**2.00));
    Vacalc=(ip*((Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + Cp**2.00 + 2.00*Cp*Cs + Cs**2.00)/(C0**2.00*Cp**2.00*Cs**2.00*Rp**2.00*om**2.00 + C0**2.00*Cp**2.00 + 2.00*C0**2.00*Cp*Cs + C0**2.00*Cs**2.00 + 2.00*C0*Cp**2.00*Cs + 2.00*C0*Cp*Cs**2.00 + Cp**2.00*Cs**2.00))**(1.00/2.00))/om;
    
    gn1 = (Va-Vacalc)/1000.00
    gn2 = (P-Pcalc)
    gn3 = (Rp-Rpcalc)/1.00e5
    gn4 = (Tin-Tincalc)/300.00
    
    # Heat and mass transfer derived experssions
    U_h=1.83*vin**(0.5) #heat transfer in gas
    K=0.017*vin**(0.5) #mass transfer in gas 
    h=50.0*vin**0.8 #heat transfer between gas and surface
    
    wAinf=0.0
    n=1.0
    
    cp1=w1*propHe['cp']+(1.0-w1)*propAir['cp'] #specific heat 
    rho1=(Patm/(R))*(propHe['Mw']*w1+propAir['Mw']*(1.0-w1))/T1 #denisty 
    
    #cp2=w2*propHe['cp']+(1.0-w2)*propAir['cp'] #specific heat 
    #rho2=(Patm/(R))*(propHe['Mw']*w2+propAir['Mw']*(1.0-w2))/T2 #denisty 
    
    #cp3=w3*propHe['cp']+(1.0-w3)*propAir['cp'] #specific heat 
    #rho3=(Patm/(R))*(propHe['Mw']*w3+propAir['Mw']*(1.0-w3))/T3 #denisty 
    
    ## differential expressions
    
    #temeprature
    dT1dt=(1.00/(rho1*cp1*(dsep/n)*dim['Ac']))*(rhoin*vin*cp1*dim['Ac']*(Tin-T1)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T1-Tinf));
    #dT2dt=(1.00/(rho2*cp2*(dsep/n)*dim['Ac']))*(rhoin*vin*cp2*dim['Ac']*(T1-T2)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T2-Tinf));
    #dT3dt=(1.00/(rho3*cp3*(dsep/n)*dim['Ac']))*(rhoin*vin*cp3*dim['Ac']*(T2-T3)-U_h*(dsep/n)*(2.00*3.1416*dim['r'])*(T3-Tinf));
    
    #mass fraction
    dw1dt=(1.00/(rho1*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(win-w1)-K*(dsep/n)*rho1*(2.00*3.1416*dim['r'])*(w1-wAinf));
    #dw2dt=(1.00/(rho2*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(w1-w2)-K*(dsep/n)*rho2*(2.00*3.1416*dim['r'])*(w2-wAinf));
    #dw3dt=(1.00/(rho3*(dsep/n)*dim['Ac']))*(rhoin*vin*dim['Ac']*(w2-w3)-K*(dsep/n)*rho3*(2.00*3.1416*dim['r'])*(w3-wAinf));
    
    #substrate
    dTs_maxdt=(dim['Ac']*h*(T1-Ts_max)-2.0*3.1416*propSurf['d']*propSurf['k']*(Ts_max-Ts_2))/(propSurf['rho']*propSurf['cp']*dim['Ac']*propSurf['d']);
    dTs2_dt=((2.0*3.1416*propSurf['d']*propSurf['k']*(Ts_max-Ts_2)-(4.0/vin)*(4.00e-3/dsep)*3.1416*propSurf['d']*propSurf['k']*(Ts_2-Tinf))/(propSurf['rho']*propSurf['cp']*dim['Ac']*propSurf['d']*100.0));
    #dCEMT = ((9.76e-14/60)*exp(0.6964*(Ts_max-273.0)))/10.0
    dCEMT = ((9.74e-14/60)*exp(0.6964*(Ts_max-273.0)))/10.0
    dCEMP = 0.001/(1e4*dim['Ac'])*1.875*exp(0.1439*(1-eta)*P)/10

    # collate expressions and create evaluator function f and integrator I
    #xdot = vertcat(dT1dt/300.0,dT2dt/300.0,dT3dt/300.0,dw1dt,dw2dt,dw3dt,dTs_maxdt/300.0,dTs2_dt/300.0)
    xdot_ss = vertcat(dT1dt/300.0,dw1dt,dTs_maxdt/300.0,dTs2_dt/300.0)
    xdot = vertcat(dT1dt/300.0,dw1dt,dTs_maxdt/300.0,dTs2_dt/300.0, dCEMT, dCEMP)
    gn   = vertcat (gn1,gn2,gn3,gn4)
    jet_dae = {'x':x, 'p':vertcat(u,d), 'z':z, 'alg':gn, 'ode':xdot}
    opts = {'tf':4}
    
    I = integrator('I', 'idas', jet_dae, opts)
    
    return [x, z, u, d, v, xdot, gn, I]
    
def dose_model(delta):
    
     #system dimensions
    dim={'r':1.5e-3};
    dim['vol']=3.1416*1e-2*dim['r']**2 #m^3 volume of plasma chamber 
    dim['Ac']=3.1416*dim['r']**2 #m^2 flow crossectional area
   
    y=SX.sym('y',2)
    u=SX.sym('u',2)
    Ts_max=u[0]*300
    P=u[1]
    
    eta= 0.4+0.07 #power deposition efficiency
    
    dCEMT = ((9.74e-14/60)*exp(0.6964*(Ts_max-273.0)))/10.0
    dCEMP = 0.001/(1e4*dim['Ac'])*1.875*exp(0.1439*(1-eta)*P)/10

    ydot = vertcat(dCEMT, dCEMP)
    dose_ode = {'x':y, 'p':u, 'ode':ydot}
    opts = {'tf':delta}
    
    I_y = integrator('I', 'idas', dose_ode, opts)
    
    return I_y