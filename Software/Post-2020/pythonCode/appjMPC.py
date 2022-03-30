##################################################################################################################################################
# This script implements a nominal model predictive controller on the atmospheric pressure plasma jet
# Inputs:
# * Sequence of applied power (u1)
# * Sequence of Helium flow rate (u2) 
#
# Outputs:
# * Minimum and maximum surface temperatures
# * Optical intensity spectrum
#
# REQUIREMENTS:
# * Python 3
# * APPJPythonFunctions.py
##################################################################################################################################################


# Some conventions
# Classes are conventionally defined with a capital first letter
# Constants are conventionally defined with all capitals


##################################################################################################################################################
# SET UP THE SCRIPT
##################################################################################################################################################
# Import libraries
import os
import sys
import matplotlib  
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import math
import warnings
import argparse
import time
import subprocess
import serial
import cv2
from seabreeze.spectrometers import Spectrometer, list_devices
from applescript import tell
from casadi import *
from datetime import datetime
from APPJPythonFunctions import*


print("--------------------------------------------------------")
# Generate timestamp
timeStamp = datetime.now().strftime('%Y_%m_%d_%H'+'h%M''m%S'+'s')

'''
# Set up the communication with the Arduino
arduinoAddress = '/dev/cu.usbmodem1434401'
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400,timeout=1)

# Obtain the spectrometer object
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])

# Obtain thermal camera object
dev, ctx = openThermalCamera()
'''


##################################################################################################################################################
# DEFINE CLASSES
##################################################################################################################################################

# Parameters
class Parameters:
    # The __init__() function is like a constructor 
    #(can give it arguments if we want to intialize some parameters depending on how we instantiate the class)
    def __init__(self): 

        # Constants (conventionally with all-capitals)
        self.TSAMP = 0.5
        self.DUTYCYCLE = 100

        # Linear state-space model !! AB: NEEDS TO BE SUBSTITUTED WITH THE IDENTIFIED LTI PLASMA MODEL
        self.A = np.array([[0.8, 0], [0, 0.2]])
        self.B = np.array([[1, 0], [0, 1]])
        self.C = np.array([[1, 0], [0, 1]])

        # Cost matrices and variable sizes
        self.nx = 2
        self.nu = 2

        # Cost matrices  # !! AB: NEEDS TO BE TUNED
        self.Q = np.array([[1, 0], [0, 0.1]])
        self.R = np.array([[0.1, 0], [0, 0.1]])
        self.Pf = np.array([[1, 0], [0, 1]])

        # Setpoint
        self.xSet = np.array([[5], [0]]).reshape(self.nx, 1)
        self.uSet = np.array([[0], [0]]).reshape(self.nu, 1)

        # Horizon lengths
        self.Np =  10     		      # Prediction horizon
        self.Nexp = 120     		  # Experiment time (1 min)

        # Variable bounds (as a list, not a numpy array) # !! AB: RECALL, THIS IS IN DEVIATION VARIABLES
        self.xMin = [-10, -10] 
        self.xMax = [10, 10]
        self.uMin = [-5, -5]
        self.uMax = [5, 5]

        # Initial state vectors (as a list, not a numpy array)
        '''
        Tmax, Tmin = getSurfaceTemperature()
        intensitySpectrum=spec.intensities()/NORMALIZATION  #NORMALIZATION is defined in APPJPuthonFunctions.py
        totalIntensity = sum(intensitySpectrum)
        '''
        # !! AB: DUMMY INITIALIZATION TO TEST THE CODE
        Tmax = 0
        totalIntensity = 0
        
        x0 = [Tmax, totalIntensity]
        self.xInit =x0


# Variables in Casasdi format
class CasadiVars:
    def __init__(self, p):
        self.x = MX.sym("x", p.nx)
        self.u = MX.sym("u", p.nu)


# Functions in Casadi format
class CasadiFunctions:
    def __init__(self, p, v):  # p is the Parameters object and v the CasadiVars object
        
        # APPJ dynamics
        xNext = mtimes(p.A, v.x)+mtimes(p.B, v.u)
        # y = mtimes(p.C, v.x)
        
        # Stage cost
        Lstage = mtimes(mtimes((v.x-p.xSet).T, p.Q), (v.x-p.xSet)) + mtimes(mtimes((v.u-p.uSet).T, p.R), (v.u-p.uSet))
                               
        # Define CasADi functions
        self.dynamics = Function('dynamics', [v.u, v.x], [xNext], ['u', 'x'], ['xNext'])
        self.stageCost = Function('stageCost', [v.u, v.x], [Lstage], ['u', 'x'], ['Lstage'])
        

# Class that defines the optimal control problem and solves the MPC
class MPC:
    
    class BuildOCP:
        def __init__(self, Parameters, CasadiVars, CasadiFunctions):
            # Start with an empty NLP
            self.w=[]    #Array of all the variables we will be optimizing over
            self.w0 = []
            self.lbw = []
            self.ubw = []
            self.J = 0
            self.g=[]
            self.lbg = []
            self.ubg = []

            # "Lift" initial conditions
            Xk = MX.sym('X0', Parameters.nx)
            self.w += [Xk]
            self.lbw += Parameters.xInit
            self.ubw += Parameters.xInit
            self.w0  += Parameters.xInit

            for i in range(0, Parameters.Np):
                # New NLP variable for the control inputs
                Uk = MX.sym('U_' + str(i), Parameters.nu)
                self.w   += [Uk]
                self.lbw += Parameters.uMin
                self.ubw += Parameters.uMax
                self.w0  += [0]*Parameters.nu
        
                # Integrate model and calculate stage cost
                xkNext = CasadiFunctions.dynamics(Uk, Xk)
                JNext = CasadiFunctions.stageCost(Uk, Xk)
                self.J = self.J + JNext

                # New NLP variable for states at the next time-step
                Xk = MX.sym('X_' + str(i+1), Parameters.nx)
                self.w   += [Xk]
                self.lbw += Parameters.xMin
                self.ubw += Parameters.xMax
                self.w0  += [0]*Parameters.nx

                # Equality constraints (model dynamics)
                self.g   += [xkNext-Xk]
                self.lbg += [0]*Parameters.nx
                self.ubg += [0]*Parameters.nx


            # Terminal cost and constraints (if applicable)
            # N/A

            # Create NLP solver
            self.prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g)}
            self.sol_opts = {'ipopt.print_level':0, 'ipopt.limited_memory_update_type':'BFGS', 'ipopt.tol':1e-4, 'ipopt.max_cpu_time':5}
            self.solver = nlpsol('solver', 'ipopt', self.prob, self.sol_opts)    

            # Store variable dimensions for easy indexing
            self.offsetX0 = 0
            self.offsetU = Parameters.nx
            self.offsetOCP = Parameters.nx + Parameters.nu

    
    
    class SolveMPC:
        def __init__(self, x0, Parameters, CasadiVars, CasadiFunctions, BuildOCP):
            
            # Predicted variables
            self.xOptPred = np.zeros((Parameters.nx, Parameters.Np+1))
            self.uOptPred = np.zeros((Parameters.nu, Parameters.Np))

            # Real variables
            self.xOptReal = np.zeros((Parameters.nx, Parameters.Nexp+1))
            self.uOptReal = np.zeros((Parameters.nu,Parameters.Nexp))
            self.stageCostReal = np.zeros((Parameters.Nexp,1))
            self.feasibility = []

            # Assign initial conditions
            self.costFnReal = 0;
            self.xOptReal = np.zeros((Parameters.nx,Parameters.Nexp+1))
            self.xOptReal[:,0] = np.array(Parameters.xInit).reshape(Parameters.nx,)
            
            for k in range(0, Parameters.Nexp):

                # Update OCP constraints on initial state
                BuildOCP.lbw[BuildOCP.offsetX0:BuildOCP.offsetX0+Parameters.nx] = x0
                BuildOCP.ubw[BuildOCP.offsetX0:BuildOCP.offsetX0+Parameters.nx] = x0
                BuildOCP.w0[BuildOCP.offsetX0:BuildOCP.offsetX0+Parameters.nx] = x0


                # Solve the NLP
                sol = BuildOCP.solver(x0=BuildOCP.w0, lbx=BuildOCP.lbw, ubx=BuildOCP.ubw, lbg=BuildOCP.lbg, ubg=BuildOCP.ubg)
                w_opt = sol['x'].full().flatten()
                J_opt = sol['f'].full().flatten()

                # Extract first optimal input
                uOpt = w_opt[BuildOCP.offsetU:BuildOCP.offsetOCP]
                # Calculate the real optimal cost
                self.stageCostReal[k] = CasadiFunctions.stageCost(uOpt, x0);

                # Plant model  !! AB: NEEDS TO BE SUBSTITUTED WITH MEASUREMENTS
                xplant = fn.dynamics(uOpt, x0) + np.random.normal(0, 0.2, size=(Parameters.nx,1))

                # Save the trajectory
                self.uOptReal[:,k] = uOpt
                self.xOptReal[:,k+1] = np.array(xplant).reshape(Parameters.nx,)


                # Determine Feasibility
                if(BuildOCP.solver.stats()['return_status']=='Infeasible_Problem_Detected'):
                    self.feasibility+=[0]
                  #  sys.exit("Infeasibility detected. Exiting")

                elif(BuildOCP.solver.stats()['return_status']=='Solve_Succeeded'):
                    self.feasibility+=[1]
                else:
                    self.feasibility+=[2]    

                print('--------------------------------------------')
                print("Iteration %i of %i" %(k+1, Parameters.Nexp))
                print('uOpt = ', uOpt)
                print(BuildOCP.solver.stats()['return_status'])
                print('--------------------------------------------')


##################################################################################################################################################
# INSTANTIATE OBJECTS
##################################################################################################################################################
# Instantiate objects
p = Parameters()
v = CasadiVars(p)
fn = CasadiFunctions(p, v)

# Create MPC object
mpc = MPC()
# Build the optimal control problem
ocp = mpc.BuildOCP(p, v, fn)
# Define the initial condition
x0 = p.xInit
# Solve the MPC
mpcSol = mpc.SolveMPC(x0, p, v, fn, ocp)

print(mpcSol.xOptReal[0,:])

plt.figure()
plt.plot(mpcSol.xOptReal[0,:])
plt.plot(mpcSol.xOptReal[1,:])
plt.show()


print('\n')