#This Script is used to generate actigraphy features and metrics for Analysis 

#Load Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import skdh #Scikit-Digital-Health for pip install
import Smoothing_Functions

#Calculates area under the curve of some input signal
def calc_area_under_curve(x,y):
    """
        :param x: x data
        :param y: y data
        :return: Area under the curve (an integration of the data)
        """
    return np.trapz(y, x, dx=.005, axis=-1)


#Scipy Peak Finder Function
def Scipy_find_peaks(x):
    """
        :param x: data to find peaks of
        :return: Plot of peaks generated, number of peaks returned
    """
    #Prominence is most important parameter, but others should be explored
    peaks, _ = scipy.signal.find_peaks(x, prominence=1) 
    plt.plot(peaks, x[peaks], "ob");
    print(f'number of peaks for prominence parameter is {len(peaks)}')
    plt.plot(x);
    plt.show()
    return len(peaks)

#Generate MAD data...To Do: Double check this with the scipy version
def VecMag_MAD(signal,wlen = 100):
    """
    :param signal: Vector Magnitude Data to compute MAD for
    :param wlen: Length of window to generate a MAD data point for
    :return: MAD data, has dimension len(signal)/wlen
    """
    r_avg = []
    MAD = []
    
    for i in range(int(len(signal)/wlen)):
        #r_avg takes a window length and computes the average in that window of the vector magnitudes.'
        #It is not a rolling average, and the end point of one window is the start point of the next. 
        #It then repeats that average a window length number of times so that the raw data and r_avg data has the same number of points. 
        #Ex) my array is [0,1,2,3,4,5,6,7,8] and wlen = 3, r_avg = [1,1,1,4,4,4,7,7,7]
        slice = signal[i*wlen:(i+1)*wlen]
        Mymean = np.mean(slice)
        temp_r_avg = np.repeat(Mymean, wlen)
        #MAD is the average of the absolute value of each vector magnitude sample minus the mean in each window
        toMAD = (abs(slice-temp_r_avg))
        MAD.append(np.mean(toMAD))

    return MAD

##Generated MAD data from triaxial raw data 
def skdh_MAD(sliced_data,wlen = 100):
    shape = (len(sliced_data['X']), 3)
    arr = np.ones(shape)
    arr[0:,0] = sliced_data['X']
    arr[0:, 1] = sliced_data['Y']
    arr[0:, 2] = sliced_data['Z']
    accel = arr
    MAD = skdh.activity.metric_mad(accel, wlen)
    myX = []
    for i in range(len(MAD)):
        myX.append(i)
    return myX, MAD

#Calculates simple signal to noise ratio
def Signal_To_Noise_Ratio(signal):
    std = np.std(signal)
    mean = np.mean(signal)
    return mean/std
