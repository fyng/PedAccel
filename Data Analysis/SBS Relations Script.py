#Import Necessary Libraries
import pandas as pd
import General_Functions
import numpy as np
import Smoothing_Functions
import Frequency_Domain
from matplotlib import pyplot as plt
import Actigraph_Metrics
import random
import statistics
import tsfresh
from scipy import stats
import time

print("This file __name__ is set to: {}" .format(__name__))

if __name__ == "__main__":

#Import sleep data,sliced in R
    RawData = pd.read_csv("Patient9MovementAnalysis/SBSneg1_num5.csv")


#Create New Columns for transitioning data
    General_Functions.create_time_column(RawData)
    General_Functions.create_absMag_Column(RawData)

#Rolling Average filter
    myData = Smoothing_Functions.moving_average(RawData, RawData['VecMag'], 11)
#Rolling average Data and raw data plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
    # create 2 subplots
    ax[0].plot(RawData['Time(s)'], RawData['VecMag'], color = 'blue')
    ax[1].plot(myData['Time(s)'], myData['averaged'], color = 'red')
    #title the plots
    ax[0].set_title('Raw Data')
    ax[0].set_ylim([0, 3])
    ax[0].set_xlabel('Time in seconds')
    ax[1].set_xlabel('Time in seconds')
    ax[0].set_ylabel('Acceleration in gs')
    ax[1].set_ylabel('Acceleration in gs')
    ax[1].set_ylim([0, 3])
    ax[1].set_title('Rolling averaged data')
    fig.suptitle('Vector magnitudes')
    #SBS in middle of a 10  minute clip
    ax[0].axvline(x= 300,color = 'black')
    ax[1].axvline(x= 300, color = 'black')
    plt.show()


#FFT
    Frequency_Domain.FFT(myData['averaged'],100,lower_freq=.3,upper_freq=20,lower_cutoff=.3,upper_cutoff = 100000, title = 'FFT of Rollig Average Data')

#MAD Plot
    myX, MAD = Actigraph_Metrics.MAD(myData, wlen=50)
    plt.plot(myX,MAD)
    plt.title('MAD plot for Raw data')
    plt.xlabel('number of samples')
    plt.ylabel('Acceleration in gs')
    plt.axvline(x=600, color = 'black')
    plt.show()

#Find peaks, area under curve, standard deviation, other metrics from Dr.Durr reccomended package tsfresh
    std = statistics.stdev(myData['averaged'])
    print(f'the standard deviation is {std}')
    area_under_curve = Actigraph_Metrics.calc_area_under_curve(x = myX, y = MAD)
    print(f'The area under the curve of rolling average of MAD data is {area_under_curve}')
    num_above = tsfresh.feature_extraction.feature_calculators.count_above(myData['averaged'], 1.1)
    print(f'The percentage of values above 1.1g is  {num_above}')
    num_below = tsfresh.feature_extraction.feature_calculators.count_below(myData['averaged'], .9)
    print(f'The percentage of values below .9g is  {num_below}')
    num_peaks = tsfresh.feature_extraction.feature_calculators.number_peaks(list(myData['averaged']), 150)
    print(f'The number of peaks in the data set where a value is greater than 150 samples to its left and right is {num_peaks}')

#Copy and paste metrics in this 5 seconds
    time.sleep(10)

#Raw and MAD data plot after thresholding for no movement
    arr = myData.copy()
    for i,v in enumerate(list(arr['averaged'])):
        if v < 1.1 and v > .9:
            arr['averaged'][i] = 1
    plt.plot(arr['Time(s)'], arr['averaged'])
    plt.xlabel('Time(seconds)')
    plt.ylabel('Acceleration in gs')
    plt.title('Thresholded plot of rolling average data')
    plt.axvline(x=300, color = 'black')
    plt.show()

#FFT of threshold and rolling average data, thresholding should remove noise
    Frequency_Domain.FFT(arr['averaged'],100,lower_freq=.3,upper_freq=20,lower_cutoff=.3,upper_cutoff = 100000, title = 'FFT of Thresholded Rollig Average Data')

#For threshold data

#Find peaks, area under curve, standard deviation, other metrics from Dr.Durr reccomended package tsfresh
    std = statistics.stdev(arr['averaged'])
    print(f'the standard deviation of threshold data is {std}')
    num_above = tsfresh.feature_extraction.feature_calculators.count_above(arr['averaged'], 1.1)
    num_peaks = tsfresh.feature_extraction.feature_calculators.number_peaks(list(arr['averaged']), 150)
    print(f'The number of peaks in the threshold data set where a value is greater than 150 samples to its left and right is {num_peaks}')