##################################################################################################################################################
# This script plots data from data collection of the APPJ
# Inputs:
# *
#
# Outputs:
# *
#
# REQUIREMENTS:
# * Python 3
# * APPJPythonFunctions.py
##################################################################################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import APPJPythonFunctions as APPJ

################################################################################################################################
# data retrieval
################################################################################################################################
pwd = os.getcwd()
# save directory in relation to current working directory
saveDirectory = '/ExperimentalData/2021_04_06_19h22m39s/'
timeStamp = saveDirectory[-21:-1]
# saveFile = '2021_03_29_15h34m24s_dataCollectionOL.csv'
# saveFile = '2021_03_29_15h34m24s_dataCollectionSpectra.csv'
# saveFile = '2021_03_29_15h34m24s_dataCollectionOscilloscope.csv'
# saveFile = '2021_03_29_15h34m24s_dataCollectionEmbedded.csv'
# saveFile = '2021_03_29_15h34m24s_dataCollectionSpatialTemps.csv'
saveFigs = 1 # 1 to save figures, otherwise don't save (Note: spectra plotting does not have this functionality)

fileType = int(input('Choose a file type:\n 0 for any arbitrary csv file\n 1 for *_dataCollectionOL\n 2 for *_dataCollectionSpectra\n 3 for *_dataCollectionOscilloscope\n 4 for *_dataCollectionEmbedded\n 5 for *_dataCollectionSpatialTemps\n'))
print("You selected: {}\n".format(fileType))
fileTypes = ['_dataCollectionOL.csv', '_dataCollectionSpectra.csv', '_dataCollectionOscilloscope.csv', '_dataCollectionEmbedded.csv', '_dataCollectionSpatialTemps.csv']
if fileType == 0:
    saveFile = input('Please input the filename you''d like to plot data in {}'.format(saveDirectory))
else:
    saveFile = timeStamp+fileTypes[fileType-1]

filePath = pwd+saveDirectory+saveFile
print("Your file is: {}\n".format(saveFile))
################################################################################################################################
# plotting functions
################################################################################################################################
def plotSimple(filePath):
    '''
    plots the data in the desired csv file using pandas dataframe plotting tool
    '''
    df = pd.read_csv(filePath)
    df.plot()
    plt.show()

def plotDataOL(filePath, saveDirectory=None):
    '''
    plots data collected in *_dataCollectionOL.csv files
    '''
    # read csv into a pandas dataframe; re-labels the columns in case of mislabels
    df = pd.read_csv(filePath, header=0, names=['Ts (degC)', 'I (a.u.)', 'P (W)', 'q (slm)'])

    Tsdata = df["Ts (degC)"]
    Isdata = df["I (a.u.)"]
    Pdata = df["P (W)"]
    qdata = df["q (slm)"]

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(Tsdata)
    ax.set_ylabel("Surface Temperature (^\circ C)")
    ax.set_title("Outputs")

    ax = fig.add_subplot(2,1,2)
    ax.plot(Isdata)
    ax.set_ylabel("Surface Intensity (a.u.)")
    ax.set_xlabel("Iteration")
    plt.draw()
    plt.show()
    if saveDirectory is not None:
        fig.savefig(saveDirectory+"dataOutputs.png")

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(Pdata)
    ax.set_ylabel("Power (W)")
    ax.set_title("Inputs")

    ax = fig.add_subplot(2,1,2)
    ax.plot(qdata)
    ax.set_ylabel("Flow Rate (slm)")
    ax.set_xlabel("Iteration")
    plt.draw()
    plt.show()
    if saveDirectory is not None:
        fig.savefig(saveDirectory+"dataInputs.png")

def plotSpectra(filePath, pauseTime=1):
    '''
    plots data collected in *_dataCollectionSpectra.csv files
    '''
    # skip the first row (comments) as well as the meanShift rows
    df = pd.read_csv(filePath, header=None, skiprows=lambda x: x%3 == 0)
    # print(df)
    wavelengths = df.iloc[0::2]
    # print(wavelengths)
    spectra = df.iloc[1::2]
    # print(spectra)
    plt.ion() # turn interactive mode on
    for i, row in wavelengths.iterrows():
        plt.plot(row, spectra.iloc[i])
        plt.draw()
        plt.show()
        plt.pause(pauseTime)
        plt.clf()

def plotOscilloscope(filePath, saveDirectory=None):
    '''
    plots data collected in *_dataCollectionOscilloscope.csv files
    '''
    # read csv into a pandas dataframe; re-labels the columns in case of mislabels
    df = pd.read_csv(filePath, header=0, names=['Vrms (V)', 'Irms (A)', 'Prms (W)'])

    # find and remove rows where data collection failed to obtain a proper reading
    df = df[df['Irms (A)'] < 1e3]

    VrmsData = df['Vrms (V)']
    IrmsData = df['Irms (A)']
    PrmsData = df['Prms (W)']

    fig = plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax.plot(VrmsData)
    ax.set_ylabel("Vrms (V)")
    ax.set_title("Oscilloscope Data")

    ax = fig.add_subplot(3,1,2)
    ax.plot(IrmsData)
    ax.set_ylabel("Irms (A)")

    ax = fig.add_subplot(3,1,3)
    ax.plot(PrmsData)
    ax.set_ylabel("Prms (W)")
    ax.set_xlabel("Iteration")
    plt.draw()
    plt.show()
    if saveDirectory is not None:
        fig.savefig(saveDirectory+"dataOscilloscope.png")

def plotIndividual(df, saveDirectory=None):
    '''
    helper function for the plotEmbedded() method. This method plots each column
    of a dataframe on its own plot

    Inputs:
    df              dataframe to be iterated over and plotted
    saveDirectory   location for which to save the plotted data

    Outputs:
    None
    '''
    for col in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(df[col])
        ax.set_ylabel(col)
        ax.set_xlabel("Iteration")
        plt.draw()
        plt.show()
        if saveDirectory is not None:
            fig.savefig(saveDirectory+"dataEmbedded"+col+".png")

def plotEmbedded(filePath, saveDirectory=None):
    '''
    plots data collected in *_dataCollectionEmbedded.csv files

    *limited functionality*
    '''
    # read csv into a pandas dataframe; re-labels the columns in case of mislabels
    df = pd.read_csv(filePath, header=0, names=['t_emb (ms)', 'Isemb (a.u.)', 'Vp2p (V)', 'f (kHz)', 'q (slm)', 'x_pos (mm)', 'y_pos (mm)', 'dsep (mm)', 'T_emb (K)', 'P_emb (W)', 'Pset (W)', 'duty (%)', 'V_emb (kV)', 'I_emb (mA)'])
    print("These are the columns read from your file:")
    print(df.columns.to_list())
    plotCols = []
    dropCols = []
    for col in df.columns:
        includeCol = input('Would you like to plot this column: {}? (y/n)\n'.format(col))
        if includeCol == 'y':
            plotCols += [col]
        elif includeCol == 'n':
            dropCols += [col]
        else:
            print('Invalid input. Will not plot this column.')
            dropCols += [col]

    df_plot = df.drop(columns=dropCols)
    # print(df_plot)
    groupData = input('Would you like to group any columns? (y/n)\n')
    if groupData == 'y':
        plottedCols = []
        while groupData == 'y':
            print('These are the columns you selected for plotting:')
            print(df_plot.columns.to_list())
            groupedCols = input('Please input the columns you would like to group with zero-indexing separated by commas (e.g. 0,3,4): \n')
            groupedColsList = [plotCols[int(i)] for i in groupedCols.split(',')]
            # print(groupedColsList)
            n = len(groupedColsList)
            fig = plt.figure()
            for i, col in enumerate(groupedColsList, start=1):
                ax = fig.add_subplot(n,1,i)
                ax.plot(df_plot[col])
                ax.set_ylabel(col)
                ax.set_xlabel("Iteration")
            plt.draw()
            plt.show()
            if saveDirectory is not None:
                plotInfo = input('Please input an additional descriptor for the data in this plot to be included in the filename. (e.g. Elec for electrical measurements)\n')
                fig.savefig(saveDirectory+"dataEmbedded"+plotInfo+".png")
                print("Figure saved under: "+saveDirectory+"dataEmbedded"+plotInfo+".png")

            groupData = input('Would you like to group any more columns? (y/n)\n')
            plottedCols += groupedColsList
        print('Plotting remaining selected columns...')
        df_plot = df_plot.drop(columns=list(set(plottedCols)))
        plotIndividual(df_plot, saveDirectory=saveDirectory)

    else:
        print('You selected no grouping. Each column of data will be plotted separately.')
        plotIndividual(df_plot, saveDirectory=saveDirectory)




################################################################################################################################
# main script
################################################################################################################################
print("Plotting data; see figures for plots. Multiple plots will show sequentially. You must close a figure before a new one is shown.")
if fileType == 0:
    plotSimple(filePath)
elif fileType == 1:
    if saveFigs == 1:
        plotDataOL(filePath, pwd+saveDirectory)
    else:
        plotDataOL(filePath)
elif fileType == 2:
    print("Optical Emission Spectra file selected...spectra will be plotted in a loop.")
    print("The default pause time between each iteration is 1 second.")
    pauseTime = input("If you desire a different pause time, enter a new pause time, otherwise press RETURN/ENTER:\n")
    if pauseTime == "":
        plotSpectra(filePath)
    else:
        plotSpectra(filePath, pauseTime=float(pauseTime))
elif fileType == 3:
    if saveFigs == 1:
        plotOscilloscope(filePath, pwd+saveDirectory)
    else:
        plotOscilloscope(filePath)
elif fileType == 4:
    if saveFigs == 1:
        plotEmbedded(filePath, pwd+saveDirectory)
    else:
        plotEmbedded(filePath)
elif fileType == 5:
    # TODO:
    pass
else:
    print("Invalid Input. Using the simple plot function.")
    plotSimple(filePath)
