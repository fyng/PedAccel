#Import Necessary Libraries
from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, sosfilt, sosfreqz
import seaborn as sns
import scaleogram as scg
import pywt #installed package name is pywavelets


def butter_bandpass(lowcut, highcut, sr, order=20):
    """
    :param lowcut: Lower cutoff frequency
    :param highcut: Upper cutoff frequency
    :param sr: sampling rate
    :param order: higher order means sharper boundaries at cutoff frequencies
    :return: Creates a new column with filtered data
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, sr=100, order=5):
    """
    :param myData: Current Pandas Data Frame
    :param data: Column of data to filter
    :param lowcut: Lower cutoff frequency
    :param highcut: Upper cutoff frequency
    :param sr: sampling frequency
    :param order: higher order means sharper boundaries at cutoff frequencies
    :return: updates column in myData with filtered data
    """
    sos = butter_bandpass(lowcut, highcut, sr, order=order)
    #myData['Butter_Bandpass_Data'] = sosfilt(sos, data)
    return sosfilt(sos, data)
def FFT(y,sr=100,lower_freq=0, upper_freq=20, lower_cutoff=40, upper_cutoff=10000,title='FFT Plot'):
    """
    :param myData: gt3x file containing accelerometry data
    :param yData_name: Column to compute the Fourier Transfrom of
    :param sr: Sampling rate, default to 100 Hz
    :param lower_freq: Minimum Frequency to Plot, default to 0Hz
    :param upper_freq: Maximum frequency to Plot, default to 20Hz
    :param lower_cutoff: Minimum Magnitude to Plot, default to 1000
    :param upper_cutoff: Maximum Magnitude to plot, default to 20000
    :return: Plot of frequency and magnitudes
    """
    input_data = np.array(y)
    X = fft(input_data)
    N = len(X)
    n = np.arange(N)
    new_array = np.abs(X)
    #create an array with the proper frequencies
    T = N / sr
    freq = n / T
    upper_freq_index = int(upper_freq*T)
    lower_freq_index = int(lower_freq*T)
    # Band pass filter with lower cutoff and upper cutoff
    new_array[new_array > upper_cutoff] = 0
    new_array[new_array < lower_cutoff] = 0
    # Get the one-sided specturm
    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    #Plot frequency decomposition
    plt.figure(figsize=(12, 6))
    print(f'The maximum frequency is {(max( (v, i) for i, v in enumerate(new_array[lower_freq_index:upper_freq_index]) )[1])/T}')
    plt.plot(f_oneside[lower_freq_index:upper_freq_index], new_array[lower_freq_index:upper_freq_index])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title(title)
    plt.show()


#Discrete Wavelet Transfrom
def CWT(signal):
    """
    :param signal: Data to compute the continuous wavelet transfrom on
    :return: Plot of Wavelet Transfrom to help interpret data
    """
# choose default wavelet function for the function
    #scg.set_default_wavelet('cmor1-1.5')
    #scg.set_default_wavelet('cgau5')
    #scg.set_default_wavelet('cgau1')
    #scg.set_default_wavelet('shan0.5-2')
    #scg.set_default_wavelet('mexh')
    # Apply DWT
# Apply CWT
    coefficients, frequencies = pywt.cwt(signal, scales=np.arange(1, 128), wavelet='cmor')
# Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), aspect='auto', cmap='jet', extent=[0, len(signal)/100, 1, 128])
    plt.colorbar(label="Magnitude")
    plt.ylabel("Scale")
    plt.xlabel("Time")
    plt.title("CWT of Data")
    plt.show()

