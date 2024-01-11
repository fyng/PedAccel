#Import Necessary Libraries
import General_Functions
import Smoothing_Functions
import Frequency_Domain
import numpy as np
from pygt3x.reader import FileReader
from matplotlib import pyplot as plt
import Actigraph_Metrics
import Correlation_and_Analysis
import pandas as pd
import skdh

print("This file __name__ is set to: {}" .format(__name__))

if __name__ == "__main__":
# Read raw data and calibrate, then export to pandas data frame
    #with FileReader("patient5_Data1.gt3x") as reader:
        #was_idle_sleep_mode_used = reader.idle_sleep_mode_activated
      #sliced_data = reader.to_pandas()
    #print(sliced_data.head(5))

    sliced_data = pd.read_csv("sliced_patient_2.csv")
    #print(sliced_data.head(5))


#Create New Columns
    General_Functions.create_time_column(sliced_data)
    General_Functions.create_absMag_Column(sliced_data)

#Data to work with
    y = sliced_data['VecMag']
    x = sliced_data['Time(s)']
    sr = 100 #sampling rate

#Plot Raw Data and FFT of Raw Data
    General_Functions.plot_Data(sliced_data,'Time(s)', 'VecMag')
    Frequency_Domain.FFT(y,sr,lower_freq= .3,upper_freq=30,lower_cutoff=0,upper_cutoff = 40, title = 'FFT of Raw Data')
    area_under_curve = Actigraph_Metrics.calc_area_under_curve(x = sliced_data['Time(s)'], y = y)
    print(f'The area under the curve of raw data is {area_under_curve}')

#find peaks in raw data
    Actigraph_Metrics.find_peaks(y)


#MAD calculation of Raw Data
    MAD = Actigraph_Metrics.MAD(sliced_data, wlen = 50)
    myX = []
    for i in range(len(MAD)):
        myX.append(i)
    plt.plot(myX,MAD)
    plt.title('MAD Plot')
    plt.show()
    area_under_curve = Actigraph_Metrics.calc_area_under_curve(x = myX, y = MAD)
    print(f'The area under the curve of MAD plot is {area_under_curve}')

#Kalman Filter
    #Smoothing_Functions.kalman_filter(sliced_data, 'VecMag')


#Down Sample
   # sliced_data = Smoothing_Functions.down_sample_to50Hz(sliced_data, 'VecMag')
    #General_Functions.plot_Data(sliced_data,'Time(s)', 'VecMag')

#Down Sampled Data to work with
    #y = sliced_data['VecMag']
    #x = sliced_data['Time(s)']
    #sr = 100 #sampling rate

#Plot Spline Fit and FFT of Spline Fit
    #Smoothing_Functions.SPline_Regression(sliced_data,x,y,smooth_param = 5)
    #Frequency_Domain.FFT(sliced_data['Spline_Regression'],sr,lower_freq = .3,upper_freq = 30,lower_cutoff = 0,upper_cutoff = 100000, title = 'FFT of Spline fit Data')

#Rolling Median
    y = Smoothing_Functions.median_filter(sliced_data,3,y,11)
    plt.plot(x,y)
    plt.title('median smoothing')
    plt.show()
    Frequency_Domain.FFT(y,sr,lower_freq=.3,upper_freq=30,lower_cutoff=0,upper_cutoff = 100000, title = 'FFT of Median FIltered Data')
    area_under_curve = Actigraph_Metrics.calc_area_under_curve(x = sliced_data['Time(s)'], y = y)
    Actigraph_Metrics.find_peaks(y)
    print(f'The area under the curve of rolling median is {area_under_curve}')

#Moving Average
    new_data = Smoothing_Functions.moving_average(sliced_data, sliced_data['VecMag'], 11)
    Frequency_Domain.FFT(new_data['averaged'],sr,lower_freq=.3,upper_freq=30,lower_cutoff=0,upper_cutoff = 100000, title = 'FFT of Rollig Average Data')
    Actigraph_Metrics.find_peaks(new_data['averaged'])
    area_under_curve = Actigraph_Metrics.calc_area_under_curve(x = new_data['Time(s)'], y = new_data['averaged'])
    print(f'The area under the curve of rolling average is {area_under_curve}')


#Buttersworth Filtered Down Sampled Rolling Median
    #Frequency_Domain.butter_bandpass_filter(sliced_data, y, lowcut = .1, highcut = 6 , sr=100)
    #Frequency_Domain.FFT(sliced_data['Butter_Bandpass_Data'],sr,lower_freq = 0,upper_freq=10,lower_cutoff = 20,upper_cutoff = 300. ,title = 'FFT of Filtered Data of Rolling Median')
    #area_under_curve = Actigraph_Metrics.calc_area_under_curve(x=sliced_data['Time(s)'], y = np.abs(np.array(sliced_data['Butter_Bandpass_Data'])))
    #print(f'The area under the curve of bandpass rolling average is {area_under_curve}')
    #plt.plot(x,(np.array(sliced_data['Butter_Bandpass_Data'])))
    #plt.title('Plot of Filtered Spline Fit .1-6Hz')
    #plt.show()
