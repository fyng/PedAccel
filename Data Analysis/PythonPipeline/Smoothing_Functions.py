#Import Necessary Libraries
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import splrep, BSpline
from scipy.fft import irfft
import scaleogram as scg
import pywt #installed package name is pywavelets
import pykalman as KalmanFilter
import math
from statsmodels.nonparametric.kernel_regression import KernelReg

#Doesn't Work
def simple_ks(y,x):
    kr = KernelReg(y, x, "c")
    y_pred, y_std = kr.fit(x)
    return y_pred

def moving_average(myData, y, w):
    """
    :param myData: Data Frame to work with
    :param y: Data to smooth
    :param w: Window of rolling average
    :return: new data frame with rolling average included
    """
    averaged = np.convolve(y, np.ones(w), 'valid') / w
    new_data = myData.copy()
    new_data = new_data.iloc[:len(averaged)]
    new_data['averaged'] = averaged
    return new_data

def down_sample_to50Hz(myData, y):
    """
      :param myData: Data frame to work with
      :param y: Data to down sample
      :return: new data frame with 50Hz sampling column
      """
    y1 = np.array(myData[y])
    avgy = ((y1 + np.roll(y1, 1)) / 2.0)[1::2]
    new_data = myData.copy()
    new_data = new_data.iloc[:len(avgy)]
    new_data['50Hz'] = avgy
    return new_data

def SPline_Regression(myData, x,y,smooth_param,title='Spline Fit Curve'):
    """
    :param myData: Data frame to modify with Spline Fit Data
    :param x: X data
    :param y: Y data
    :param smooth_param: Larger smooth_param means more smoothing of the raw data
    :return: Creates a new column in myData with smoothed values
    """
    tck = splrep(x, y, s=smooth_param)
    plt.plot(x, BSpline(*tck)(x))
    plt.title(title)
    plt.show()
    myData['Spline_Regression'] = BSpline(*tck)(x)

def median_filter(myData,num_passes, myDataArray,window_size = 301):
    """
        :param myData: Data frame to modify with Spline Fit Data
        :param num_passes: number of times to smooth the data
        :param myDataArray: Y data
        :param window_size: size of window for rolling median
        :return: Creates a new column in myData with smoothed values called median_filter
        """
    for i in range(num_passes):
        myDataArray = scipy.signal.medfilt(myDataArray, kernel_size=window_size)
        myData['median_filter'] = myDataArray
    return myDataArray

#Ignore for Now
def Wavelet_Smoothing(x,y, threshold=.5):
    signal = y
    coeffs = pywt.wavedec(signal, 'db1', level=6)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
    # Plotting the noisy and denoised signals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, signal)
    plt.title("Noisy Signal")
    plt.subplot(1, 2, 2)
    plt.plot(x, denoised_signal)
    plt.title("Denoised Signal")
    plt.tight_layout()
    plt.show()


#Ignore For now
def kalman_filter(df,y_name):
    X = df.drop(y_name, axis=1)
    y = df[y_name]

    estimated_value = np.array(X)
    real_value = np.array(y)

    measurements = np.asarray(estimated_value)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=1,
                      transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=measurements[0, 1],
                      initial_state_covariance=1,
                      observation_covariance=5,
                      transition_covariance=1)

    state_means, state_covariances = kf.filter(measurements[:, 1])
    state_std = np.sqrt(state_covariances[:, 0])
    print(state_std)
    print(state_means)
    print(state_covariances)

    fig, ax = plt.subplots()
    ax.margins(x=0, y=0.05)

    plt.plot(measurements[:, 0], measurements[:, 1], '-r', label='Real Value Input')
    plt.plot(measurements[:, 0], state_means, '-b', label='Kalman-Filter')
    plt.legend(loc='best')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    plt.show()
