# FLiR-lepton with Arduino Due

### Commands

To use this code, connect the camera, open the serial monitor and use the following commands for its respective actions:

- To capture Image: captureImage
- Print the image into the serial monitor: transferImage
- To get the temperature of the camera core: getTemperature
- To find the minimum raw value of the captured Image: findMin


### Connections

In the folder connections there is an image to show how the Arduino Due has to be connected with the PureEngineering Module.


### Instructions

* **1:** When downloaded, change the folder name from FLiR-lepton-master to FLiR-lepton.

* **2:** The serial monitor has to be configured to write a new line, option located at the bottom right corner. 

* **3:** The baud rate also has to be configured to be 115200. 


### Tips

You should know that there is a checkpoint system for the actions, so when an image is captured, you should expect two "OK", if there is only one, it means that the process was not completed successfully, reset the Arduino and try again.

To visualize the thermal image, what you can do is copy the data from the serial monitor, paste it in an excel sheet and give the data a conditional coloured format.

Don't use a breadboard between the camera and the Arduino, it may cause some issues, connect it directly with wires.

Be sure that the lepton core is correctly fitted into its socket, but be careful doing it.

Let the camera running for at least 5 minutes before start using it. It will give you more constant values because the core will be heated.

### From Raw to temperature

You cannot get directly those values from the camera, first you should do a calibration. The steps I did are the following:
* **1:** Get a device where you can change the temperature and connect a thermometer, a Peltier could be useful.

* **2:** Make some readings and create a table where it should appear a comparision between the RAW value of the object and the temperature reading of the thermometer. 

* **3:** With a formula of the type (T = a·RAW + T_C - k) where T is the temperature of the object, RAW the value that the camera captured (a mid value of a region for example) and T_C is the temperature of the camera, you have to fit the formula (modifying the a and k values) with the values of the table described at the second point. The fitting has to be done with some mathematical method, but a simple option could be doing it with 2 table values, that will give us a two equation system with 2 unkown values, which will make you able to fins a value for the "a" and a value for the "k".

* **4:** Finally you will have a formula like: T = 0.03385·RAW + T_C - 276.96 and if the RAW is equal to 8152 and the T_C is 33.21, the temperature reading will be 32.19 ºC. 

**Note**: This process is not very accurate, but could give you a good approximation of the temperature, it is far to be the best way to do it, it is only an example, if you have a better one, tell me and I'll update it.

### Image Example
![alt tag](https://github.com/josepbordesjove/FLiR-lepton/blob/master/Connections/Image%204.png)

If you have any issues, please contact me at josepbordesjove@gmail.com and ask.

Also I developed a version for the Raspberry, which is more reliable, https://github.com/josepbordesjove/LeptonCapture

If you liked the project, please give it a STAR!
