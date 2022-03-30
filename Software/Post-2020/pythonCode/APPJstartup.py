import os
from applescript import tell
import time
import subprocess
from APPJPythonFunctions import getArduinoAddress

#Set directory (in case of drag-and-drop directory will not be already set)
directory = os.getcwd()
os.chdir(directory)


# Define commands here
testCommand = 'ls'
arduinoAddress = getArduinoAddress();
arduinoSetUp = 'stty -f '+ arduinoAddress +' raw 38400 -hupcl & cat '+arduinoAddress #Make sure that the address correspond that shown in the arduino IDE

# Open new terminal window to read the arduino
print("Opening new terminal window to read the arduino...")
time.sleep(1)
tell.app( 'Terminal', 'do script "' + arduinoSetUp + '"')
time.sleep(3)
print("...Done")


# Set the default values
dutyCycleIn = 100;
powerIn = 2;
flowIn = 1.5;

time.sleep(0.5)
print("Setting duty cycle to " + str(dutyCycleIn) + " %" )
os.system("echo \"p,"+str(dutyCycleIn)+"\" > "+ arduinoAddress)
time.sleep(0.5)
print("Setting power to " + str(powerIn) + " W")
os.system("echo \"w,"+str(powerIn)+"\" > " + arduinoAddress)
time.sleep(0.5)
print("Setting flow rate to " + str(flowIn) + " slm")
os.system("echo \"q,"+str(flowIn)+"\" > " + arduinoAddress)
time.sleep(0.5)

quit = False
while(quit==False):
	try:
		time.sleep(0.2)
		stringInput = input(">> Set desired values of power and flow as P,q (type quit to exit): ")
		powerIn, flowIn = stringInput.split(',')
		if(powerIn!=-1 and flowIn!=-1):
			os.system("echo \"w,"+str(powerIn)+"\" > " + arduinoAddress)
			time.sleep(0.1)
			os.system("echo \"q,"+str(flowIn)+"\" > " + arduinoAddress)
			print("Inputs sent! P = " + str(powerIn)+ " and q = " + str(flowIn))
	except:
		if(stringInput=="quit"):
			quit=True
		else:
			print("Invalid input. Pass two inputs separated by a comma.")
			pass

os.chdir(directory)
print("\n##########################################################################################################################################################################################")
print("MAKE SURE TO STOP READING THE ARDUINO FROM THE SECOND TERMINAL WINDOW BEFORE EXECUTING ANY PYTHON SCRIPTS (CNTRL+C)! YOU CANNOT READ THE ARDUINO FROM TWO DIFFERENT PLACES AT THE SAME TIME!")
print("##########################################################################################################################################################################################\n")
