#Import Necessary Libraries
from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, sosfilt, sosfreqz
import scipy
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

def FFTData(y,sr=100):
#Generates FFT Data
    input_data = np.array(y)
    X = fft(input_data)
    N = len(X)
    n = np.arange(N)
    #create an array with the proper frequency bins
    T = N / sr
    freq = n / T
    #returns complex coefficients, not magnitude
    return list(X), list(freq)

def FFTPlot(X, freq,sr=100,lower_freq=0, upper_freq=20, lower_cutoff=40, upper_cutoff=10000,title='FFT Plot'):
#Accepts FFT Data and formats it for plotting
    new_array = np.abs(X)
    N = len(X)
    T =N/sr
    #create an array with the proper frequencies
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


def average_out_harmonics(signal,f0=1,fmax=100):
#Averages out complex number every 1Hz
    freqData,freqBins = FFTData(signal, 100)
    for i in range(f0,fmax+1,f0):
        index = freqBins.index(i)
        print(f'This Frequency is being averaged out {freqBins[index]}')
        print(f'This value is being averaged out {freqData[index]}')
        freqData[index] = (np.average(freqData[index-100:index] + freqData[index+1:index+101]))
        print(f'This value is averaged out {freqData[index]}')
    FFTPlot(freqData, freqBins,sr=100,lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT with averaged out harmonics Plot')
    # returns FFT complex coefficients for inverse
    return freqData, freqBins


def remove_outliers_freqDomain(signal,wlen):
#Averages out outliers
    freqData,freqBins = FFTData(signal, 100)
    #Use magnitude of complex coefficients for outlier detection
    freqAbs = np.abs(freqData)
    Q1 = np.percentile(freqAbs, 25, interpolation='midpoint')
    Q3 = np.percentile(freqAbs, 75, interpolation='midpoint')
    #4.5 * IQR instead of 1.5
    outlier = 6 * (Q3 - Q1)
    count = 0
    for i, v in enumerate(freqData):
        if np.abs(v) > outlier and i >= wlen and i <= (len(freqData) - wlen):
            freqData[i] = np.average(freqData[i - wlen:i + wlen])
            count += 1
        elif np.abs(v) > outlier and i <= wlen:
            freqData[i] = np.average(freqData[:i + 2*wlen])
        elif np.abs(v) > outlier and i >= wlen:
            freqData[i] = np.average(freqData[i - 2*wlen:])
    print(count)
    FFTPlot(freqData, freqBins, sr=100, lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT with outliers averaged out')
    # returns FFT complex coefficients for inverse
    return freqData, freqBins

def freq_domain_rollingAvg(signal,wlen,fs=100):
    #Take rolling average in Frequency Domain
    freqData,freqBins = FFTData(signal, 100)
    averaged = np.convolve(np.array(freqData), np.ones(wlen), 'valid') / wlen
    #Rollinga average changes number of samples, re-define frequency bins
    print(averaged[:5])
    N = len(averaged)
    n = np.arange(N)
    # create an array with the proper frequencies
    T = N / fs
    freqBins = n / T
    #Plot data
    FFTPlot(averaged, freqBins, sr=100, lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT with rolling average')
    #returns averaged out FFT complex coefficients for inverse
    return averaged, freqBins

def scipy_comb_filter(signalcf, fs=100, f0=1, Q=20):
    b, a = scipy.signal.iircomb(f0, Q, ftype = 'notch', fs=fs)
    y_filter = scipy.signal.lfilter(b, a, signalcf)
    freqData, freqBins = FFTData(y_filter,100)
    FFTPlot(freqData, freqBins,sr=100,lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT with scipy comb filter')
    #returns filtered time series data
    return y_filter

def myCombFilter(signalcf, f0=1, Q=5, fmax=50,fs=100):
    #Notch filter at every 1Hz through fmax
    for i in range(1,fmax+1,f0):
        b,a = scipy.signal.iirnotch(f0,Q,fs)
        signalcf = scipy.signal.lfilter(b, a, signalcf)
        freqData, freqBins = FFTData(signalcf, 100)
    FFTPlot(freqData, freqBins, sr=100, lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT my comb filter')
    #returns filtered time series data
    return signalcf

def inverse_FFT(y):
    itx = np.fft.ifft(y)
    return itx
