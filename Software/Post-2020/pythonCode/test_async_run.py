##################################################################################################################################################
# This script tests asynchronous measurements of the APPJ
#
# REQUIREMENTS:
# * Python 3
# * APPJPythonFunctions.py
##################################################################################################################################################

# Import libraries
import os
import sys
import serial
from seabreeze.spectrometers import Spectrometer, list_devices
from APPJPythonFunctions import*

tSampling = 1.0

runOpts = RunOpts()

# Define arduino address
arduinoAddress = getArduinoAddress()

# define serial output from Arduino
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400,timeout=1)

# Oscilloscope Setup
oscilloscope = Oscilloscope()       # Instantiate object from class
instr = oscilloscope.initialize()	# Initialize oscilloscope
instr.timeout = 1

# Obtain the spectrometer object
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])
spec.integration_time_micros(12000*8)

# Obtain thermal camera object
dev, ctx = openThermalCamera()

# get event loop for asynchronous measurement
if os.name == 'nt':
    ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(ioloop)
else:
    ioloop = asyncio.get_event_loop()

# run once to initialize
prevTime = 0.0
tasks, runTime = ioloop.run_until_complete(async_measure(arduinoPI, prevTime, instr, spec, runOpts))

print('measurement devices ready!')
# ignite plasma
w = [2,2.5,1.5,2,3,1.5,2.5,4,2,2.5]
sendInputsArduino(arduinoPI, 2, 2, 100, arduinoAddress)
plasmaOn = input('press Return when plasma ignites\n')

Niter = 10
run = True
avgRunTime = 0
timeDiff = []
prevTime = tasks[3].result()[0]
for i in range(Niter):
    startTime = time.time()
    arduinoPI.reset_input_buffer()
    tasks, runTime = ioloop.run_until_complete(async_measure(arduinoPI, prevTime, instr, spec, runOpts))

    avgRunTime += runTime
    # tasks is a list of completed tasks from the function async_measure()

    # print("printing THERMAL CAMERA measurements:")
    # thermalCamMeasure = tasks[0].result()
    # Ts = thermalCamMeasure[0]
    # Ts2 = thermalCamMeasure[1]
    # Ts3 = thermalCamMeasure[2]
    # print('Surface Temperature: ', Ts)
    # print('Surface Temperature 2 px away: ', Ts2)
    # print('Surface Temperature 12 px away: ', Ts3)
    #
    # print('printing SPECTROMETER measurements:')
    # specOut = tasks[1].result()
    # print(specOut)
    #
    # print('printing OSCILLOSCOPE measurements:')
    # oscOut = tasks[2].result()
    # print(oscOut)
    #
    # print('printing ARDUINO outputs:')
    arduinoOut = tasks[3].result()
    # print(arduinoOut)
    timeDiff += [arduinoOut[0]]
    prevTime = arduinoOut[0]

    sendInputsArduino(arduinoPI, w[i], 2, 100, arduinoAddress)

    endTime = time.time()
    rTime = (endTime-startTime)
    pauseTime = tSampling - rTime
    if pauseTime < 0:
        print('WARNING: Measurement time took longer than your sampling time! Exiting measurement.')
        # Close devices
        closeThermalCamera(dev, ctx)
        instr.close()
        sys.exit(-1)
    else:
        time.sleep(pauseTime)

print('Time Difference between Embedded Measurements:')
print(np.diff(np.array(timeDiff)))
# Close devices
closeThermalCamera(dev, ctx)
instr.close()

print("Average Runtime of Data Collection: ", avgRunTime/Niter)
