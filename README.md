# Advanced Process Control for Atmospheric Pressure Plasmas

The work in this repository is done in the Chemical and Biomolecular Engineering Department at University of California, Berkeley in the [**Mesbah Lab**](http://www.mesbahlab.com/)

*This repository was last updated March 2022*

Atmospheric Pressure Plasma Jet (APPJ) devices exhibit highly variable behavior in response to changes in their environment. Small changes in, for example jet-tip-to-substrate separation distance can cause drastic changes in the APPJ behavior. Using feedback control, we can mitigate some of this variability allowing for reliable, reproducible and effective operation of APPJs. Below, you can see how feedback control can help regulate substrate temperautre while the seperation distance changes (6x sped up). 

* Left: No Control
* Middle: Voltage Controls Temperature V=>T
* Right: Flow Controls Temperature q=>T 

![Helium plasma jet](/Results/compare_pi.gif)

The maximum temperature measured at the surface for each case clearly demonstrates the need for and benefit of feedback control

![Helium plasma jet](/Results/Temp_prof.png)

For practial applications, the APPJ should also be translated over the substrate as often the APPJ sources have an effect area (~few mm^2) much smaller than practical targets (several cm^2). We achieve this using a moving stage controlled by stepper motors akin to conventional 3D printers. This allows us to freely and precisely move the APPJ in 3D.

Watch the plasma translating over a surface!

![Helium plasma jet](/Img/moving_jet.gif)

This repository contains code used for implementing classical (proportional-integral-derivative) and advanced (optimization-based) feedback control on a kHz-excited APPJ in Helium 

## Hardware
The APPJ Control project makes use of open-soruce and DIY hardware where possible. With the exception of the high voltage amplifier, all components used, including the microcontrollers, single board computer, circuitry etc. are open source and comparatively cheap. Detailed information about the used hardware can be founed under the [/Hardware](https://github.com/kchan45/APPJ_Control/tree/master/Hardware) directory, including circuit diagrams and datasheets.

## Firmware
The Arduino firmware allows exchanging commands with the Arduino via serial monitor. The latest version (as of 14/6/18) is version 14, available under [/Firmware](https://github.com/kchan45/APPJ_Control/tree/master/Firmware). The firmware is configured to manage two gas flow rates, 3D position, applied voltage and frequency. Default measurements include RMS voltage, RMS current, total emission intensity and ambient temperature.

## Software
*Beginning Summer 2020, the APPJ was moved, and some software components have changed significantly.*
Detailed information about the software is under [/Software](https://github.com/kchan45/APPJ_Control/tree/master/Software). Bulk of the software is written in Python and can be easily modified for different applications. There is a client-server architecture in place based on TCP/IP protocol. This allows information exchange with the data aquisiton software and allows for complicated calculations to be done on external computers, on the cloud etc.
*As of Summer 2020, the client-server architecture has been removed. All information exchange and data acquisition should be done on an external computer connected to the setup. Detailed information regarding this change will be added soon...*

## Original Academic Publications
* D. Gidon, B. Curtis, J. A. Paulson, D. B. Graves and A. Mesbah, ["Model-Based Feedback Control of a kHz-Excited Atmospheric Pressure Plasma Jet,"](http://doi.org/10.1109/TRPMS.2017.2764629) in IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 2, no. 2, pp. 129-137, March 2018.
doi: 10.1109/TRPMS.2017.2764629
*  D. Gidon, D. B. Graves, and A. Mesbah, [“Effective dose delivery in atmospheric pressure plasma jets for plasma medicine: A model predictive control approach,”](http://iopscience.iop.org/article/10.1088/1361-6595/aa7c5d/meta) Plasma Sources Sci. Technol., vol. 26, no. 8, pp. 85005–85019, 2017.

*This repository has been forked and modified from [**APPJ_Control**](https://github.com/dgngdn/APPJ_Control) by Dogan Gidon, and previous editions of this project were done in collaboration with the [**Graves Lab**](https://graveslab.org)*
