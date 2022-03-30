##################################################################################################################################################
# This script sends inputs to the APPJ setup and collects the respective measurements to be used for model identification.
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
# * APPJConstants.py
##################################################################################################################################################

# Import libraries
import sys
import argparse
from seabreeze.spectrometers import Spectrometer, list_devices
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from applescript import tell
import subprocess
import serial
import cv2
from datetime import datetime
from APPJPythonFunctions import*


#Generate timestamp
timeStamp = datetime.now().strftime('%Y_%m_%d_%H'+'h%M''m%S'+'s')
##################################################################################################################################################
# USER INPUTS
##################################################################################################################################################
# arduinoAddress = '/dev/cu.usbmodem14211401'
# arduinoAddress = '/dev/cu.usbmodem1434401'
arduinoAddress = getArduinoAddress()
# Move constants to APPJConstants.py file so that they are separate!

runOpts = RunOpts()
dutyCycle = 100
runOpts.tSampling = 1.0		#in seconds
tSampling = runOpts.tSampling
DeltaTimeInput = 90 #in seconds


# collection options, by default, all true
# runOpts.collectOscMeas = False
# runOpts.collectData = False
# runOpts.collectSpatialTemp = False
# runOpts.collectEntireSpectra = False
# runOpts.collectOscMeas = False
# runOpts.collectEmbedded = False
if (runOpts.collectOscMeas and tSampling < 0.8):
    print('WARNING: Oscilloscope measurements are being collected, and your sampling time may be too low!')
    exitOpt = input('Input 1 to exit or 0 to continue:\n')
    if exitOpt == 1:
        quit()
# save options, by default, all save options are True
# runOpts.saveData = False
# runOpts.saveSpatialTemp = False
# runOpts.saveSpectra = False
# runOpts.saveOscMeas = False
# runOpts.saveEmbMeas = False

# Define input sequence for model identification (u1: power; u2: flow)
# u1Vec = np.linspace(1.5, 5, 5)
# u2Vec = np.linspace(1.5, 5, 5)
# u1Vec = np.array([1.5, 2, 2.5])
# u2Vec = np.array([2, 2.5, 3])
u1Vec = np.array([2, 3])
u2Vec = np.array([1.5])
uu1, uu2 = np.meshgrid(u1Vec, u2Vec)
u1Vec = uu1.reshape(1,-1)
u2Vec = uu2.reshape(1,-1)

np.random.seed(0)
# np.random.shuffle(u1Vec.reshape(-1,))
# np.random.shuffle(u2Vec.reshape(-1,))

u1Vec = np.repeat(u1Vec, DeltaTimeInput/tSampling).reshape(1,-1)
u2Vec = np.repeat(u2Vec, DeltaTimeInput/tSampling).reshape(1,-1)


##################################################################################################################################################
# MAIN SCRIPT
##################################################################################################################################################

# Print a line to quickly spot the start of the output
print('\n\n\n\n--------------------------------------------')

# Check that the number of arguments is correct
# NUMARG = 2
# if len(sys.argv)!=NUMARG:
# 	print("Function expects "+str(NUMARG-1)+" argument(s). Example: [...]")
# 	exit()

# Define arduino address
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400,timeout=1)

# Oscilloscope Setup
oscilloscope = Oscilloscope()       # Instantiate object from class
instr = oscilloscope.initialize()	# Initialize oscilloscope

# Obtain the spectrometer object
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])
spec.integration_time_micros(12000*6)

# Obtain thermal camera object
dev, ctx = openThermalCamera()

# set up asynchronous measurement
if os.name == 'nt':
    ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(ioloop)
else:
    ioloop = asyncio.get_event_loop()
# run once to initialize
prevTime = 0.0
tasks, runTime = ioloop.run_until_complete(async_measure(arduinoPI, prevTime, instr, spec, runOpts))
print('measurement devices ready!')

s = time.time()
thermalCamOut = tasks[0].result()
specOut = tasks[1].result()
oscOut = tasks[2].result()
arduinoOut = tasks[3].result()
prevTime = arduinoOut[0]

# Initialize arrays/lists in which to save outputs
if runOpts.saveData:
    Tsave = []
    Isave = []
    badTimes = []

if runOpts.saveSpatialTemp:
    Ts2save = []
    Ts3save = []

if runOpts.saveSpectra:
    if specOut is not None:
        Ispec = np.empty((1, len(specOut[2])))
    else:
        print('Intensity Data not collected! Entire spectrum will not be saved.')
        runOpts.saveSpectra = False

if runOpts.saveOscMeas:
    if oscOut is not None:
        oscSave = np.empty((1, len(oscOut)))
    else:
        print('Oscilloscope data not collected! Nothing to save.')
        runOpts.saveOscMeas = False

if runOpts.saveEmbMeas:
    if arduinoOut is not None:
        ArdSave = np.empty((1,len(arduinoOut)))
    else:
        print('Arduino Data not collected! Nothing to save.')
        runOpts.saveEmbMeas = False

sendInputsArduino(arduinoPI, u1Vec[0,0],u2Vec[0,0],dutyCycle, arduinoAddress)
input("Measurements are ready to be recorded. Ensure plasma has ignited and press Return to begin.")
Niter = u1Vec.shape[1]
# Niter = 10
avgRunTime = 0
prevTime = (time.time()-s)*1e3
for i in range(Niter):
    startTime = time.time() # record start time of iteration
    iterString = "\nIteration %d out of %d" %(i, Niter)
    print(iterString)

    # Asynchronous measurement <--- 2021/03/24: takes on average 0.3 seconds, but varies between 0.1-0.5
    tasks, _ = ioloop.run_until_complete(async_measure(arduinoPI, prevTime, instr, spec, runOpts))

    # Temperature
    thermalCamMeasure = tasks[0].result()
    if thermalCamMeasure is not None:
        Ts = thermalCamMeasure[0]
        Ts2 = thermalCamMeasure[1]
        Ts3 = thermalCamMeasure[2]
    else:
        print('Temperature data not collected! Thermal Camera measurements will be set to -300.')
        Ts = -300
        Ts2 = -300
        Ts3 = -300

    # Total intensity
    specOut = tasks[1].result()
    if specOut is not None:
        totalIntensity = specOut[0]
        intensitySpectrum = specOut[1]
        wavelengths = specOut[2]
        meanShift = specOut[3]
    else:
        print('Intensity data not collected! Spectrometer outputs will be set to -1.')
        totalIntensity = -1
        intensitySpectrum = -1
        wavelengths = -1
        meanShift = -1

    # Save measurements <--- takes on the order of 1-2e-5 seconds
    if runOpts.saveData:
        Tsave += [Ts]
        Isave += [totalIntensity]
    if runOpts.saveSpatialTemp:
        Ts2save += [Ts2]
        Ts3save += [Ts3]
    # Intensity spectra (row 1: wavelengths; row 2: intensities; row 3: mean value used to shift spectra)
    if runOpts.saveSpectra:
        Ispec = np.append(Ispec, wavelengths.reshape(1, -1), axis=0)
        Ispec = np.append(Ispec, intensitySpectrum.reshape(1, -1), axis=0)
        Ispec = np.append(Ispec, (meanShift+np.zeros(wavelengths.shape)).reshape(1,-1), axis=0)

    # Oscilloscope
    if runOpts.saveOscMeas:
        oscOut = tasks[2].result()
        oscSave = np.append(oscSave, oscOut.reshape(1,-1), axis=0)

    # Embedded Measurements from the Arduino
    arduinoOut = tasks[3].result()
    prevTime = arduinoOut[0]
    if runOpts.saveEmbMeas:
        ArdSave = np.append(ArdSave, arduinoOut.reshape(1,-1), axis=0)

    outString = "Measured Outputs: Temperature:%.2f, Intensity:%.2f" %(Ts,totalIntensity)
    print(outString)

    # Send inputs <--- takes at least 0.15 seconds (due to programmed pauses)
    sendInputsArduino(arduinoPI, u1Vec[0,i],u2Vec[0,i],dutyCycle, arduinoAddress)

    # Pause for the duration of the sampling time to allow the system to evolve
    endTime = time.time()
    runTime = endTime-startTime
    print('Total Runtime was:', runTime)
    pauseTime = tSampling - runTime
    if pauseTime>0:
        print("Pausing for {} seconds...".format(pauseTime))
        time.sleep(pauseTime)
    else:
        print('WARNING: Measurement Time was greater than Sampling Time! Data may be inaccurate.')
        if runOpts.saveData:
            badTimes += [i]

# Close devices
closeThermalCamera(dev, ctx)
instr.close()

if runOpts.saveSpectra:
    Ispec = np.delete(Ispec, 0, 0)
if runOpts.saveOscMeas:
    oscSave = np.delete(oscSave, 0, 0)
if runOpts.saveEmbMeas:
    ArdSave = np.delete(ArdSave, 0, 0)

# Save to CSV
saveConditions = [runOpts.saveData, runOpts.saveSpatialTemp, runOpts.saveSpectra, runOpts.saveOscMeas, runOpts.saveEmbMeas]
if any(saveConditions):
    directory = os.getcwd()
    os.makedirs(directory+"/ExperimentalData/"+timeStamp, exist_ok=True)
    saveDir = directory+"/ExperimentalData/"+timeStamp+"/"
    print('\n\nData will be saved in the following directory:')
    print(saveDir)
if runOpts.saveData:
    # Concetenate inputs and outputs into one numpy array to save it as a csv
    dataHeader = "Ts (degC),I (a.u.),P (W),q (slm)"
    saveArray = np.hstack((np.array(Tsave).reshape(-1,1), np.array(Isave).reshape(-1,1), u1Vec[:,:Niter].T, u2Vec[:,:Niter].T))
    np.savetxt( saveDir+timeStamp+"_dataCollectionOL.csv", saveArray, delimiter=",", header=dataHeader, comments='')
    if badTimes:
        np.savetxt( saveDir+timeStamp+"_badMeasurementTimes.csv", badTimes, delimiter=',')
if runOpts.saveSpatialTemp:
    dataHeader = "Ts2 (degC),Ts3 (degC)"
    saveArray = np.hstack((np.array(Ts2save).reshape(-1,1), np.array(Ts3save).reshape(-1,1)))
    np.savetxt( saveDir+timeStamp+"_dataCollectionSpatialTemps.csv", saveArray, delimiter=",", header=dataHeader, comments='')
if runOpts.saveSpectra:
    dataHeader = "Each iteration is saved in 3 rows. Row 3k+2 is the wavelengths (nm); Row 3k+3 is the spectra (a.u.); Row 3k+4 is the mean shift (a.u.) defined as the mean of the last 20 points of the spectra, where k is the iteration number."
    np.savetxt( saveDir+timeStamp+"_dataCollectionSpectra.csv", Ispec, delimiter=",", header=dataHeader)
if runOpts.saveOscMeas:
    dataHeader = "Vrms (V),Irms (A),Prms (W)"
    np.savetxt( saveDir+timeStamp+"_dataCollectionOscilloscope.csv", oscSave, delimiter=",", header=dataHeader, comments='')
if runOpts.saveEmbMeas:
    dataHeader = "t_emb (ms),Isemb (a.u.),Vp2p (V),f (kHz),q (slm),x_pos (mm),y_pos (mm),dsep (mm),T_emb (K),P_emb (W),Pset (W),duty (%),V_emb (kV),I_emb (mA)"
    np.savetxt( saveDir+timeStamp+"_dataCollectionEmbedded.csv", ArdSave, delimiter=",", header=dataHeader, comments='')
