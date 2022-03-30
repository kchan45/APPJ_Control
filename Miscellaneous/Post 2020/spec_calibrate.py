import sys
sys.path.append('/home/brandon/repos/python-seabreeze')
import seabreeze.spectrometers as sb
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse

devices = sb.list_devices()
#t_int=12000s
t_int=12000*6
print("Available devices {}".format(devices))
spec = sb.Spectrometer(devices[0])
print("Using {}".format(devices[0]))
spec.integration_time_micros(t_int)



def get_runopts():
  """
  Gets the arguments provided to the interpreter at runtime
  """
  parser = argparse.ArgumentParser(description="runs MPC",
			  epilog="Example: python mpc_lin_test.py --quiet")
  parser.add_argument("--live", help="plots live data", action="store_true", default=False)
  parser.add_argument("--save_spec", help="save OES spectra", action="store_true")
  runopts = parser.parse_args()
  return runopts

## Main Things

opts=get_runopts()
if opts.live:
    plt.ion()
    while True:

        wv=spec.wavelengths()
        sp=spec.intensities()

        wv=wv[10:]
        sp=sp[10:]/np.max(sp[10:])
        print(np.sum(sp[10:]))

        plt.cla()
        plt.plot(wv,sp)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.draw()
        plt.pause(0.05)

else:

    wv=spec.wavelengths()
    sp=spec.intensities()

    wv=wv[10:]
    sp=sp[10:]/np.max(sp[10:])

    plt.cla()
    plt.plot(wv,sp)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.show()   

