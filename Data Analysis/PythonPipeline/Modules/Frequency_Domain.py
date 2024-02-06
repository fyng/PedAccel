#This Script is used for Frequency Domain analysis, digital filtering, and more

#Import Necessary Libraries
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, sosfilt, sosfreqz
import pywt #installed package name is pywavelets



def butter_bandpass(lowcut, highcut, sr, order=20):
    """
    :param lowcut: Lower cutoff frequency
    :param highcut: Upper cutoff frequency
    :param sr: sampling rate
    :param order: higher order means sharper boundaries at cutoff frequencies
    :return: Second-order sections representation of the IIR filter
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

#Bandpass filter with sharp boundaries
def butter_bandpass_filter(signal, lowcut, highcut, sr=100, order=5):
    """
    :param signal: Current signal
    :param lowcut: Lower cutoff frequency
    :param highcut: Upper cutoff frequency
    :param sr: sampling frequency
    :param order: higher order means sharper boundaries at cutoff frequencies
    :return: filtered data
    """
    sos = butter_bandpass(lowcut, highcut, sr, order=order)
    return sosfilt(sos, signal)

#Discrete Wavelet Transfrom
def CWT(signal):
    """
    :param signal: Data to compute the continuous wavelet transfrom on
    :return: Plot of Wavelet Transfrom to help interpret data
    """
# choices for other wavelets
    wavelet = 'cmor'
    #scg.set_default_wavelet('cgau5')
    #scg.set_default_wavelet('cgau1')
    #scg.set_default_wavelet('shan0.5-2')
    #scg.set_default_wavelet('mexh')
        
# Apply CWT
    coefficients, frequencies = pywt.cwt(signal, scales=np.arange(1, 128), wavelet= wavelet)

# Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), aspect='auto', cmap='jet', extent=[0, len(signal)/100, 1, 128])
    plt.colorbar(label="Magnitude")
    plt.ylabel("Scale")
    plt.xlabel("Time")
    plt.title("CWT of Data")
    plt.show()

#Generates FFT Data
def FFTData(signal,sr=100):
    """
    :param signal: Data to compute the FFT for
    :return: Complex Coefficients for FFT
    """
    input_data = np.array(signal)
    X = fft(input_data)
    N = len(X)
    n = np.arange(N)
    #create an array with the proper frequency bins
    T = N / sr
    freq = n / T
    #returns complex coefficients, not magnitude
    return list(X), list(freq)

def inverse_FFT(FFTData):
    itx = np.fft.ifft(FFTData)
    return itx

#Accepts FFT Data and formats it for plotting
def FFT(y, sr=100, lower_freq=0, upper_freq=20, lower_cutoff=40, upper_cutoff=10000, title='FFT Plot', figsize=(12, 6), xlim=None, ylim=None):
    """
    :param myData: gt3x file containing accelerometry data
    :param yData_name: Column to compute the Fourier Transfrom of
    :param sr: Sampling rate, default to 100 Hz
    :param lower_freq: Minimum Frequency to Plot, default to 0Hz
    :param upper_freq: Maximum frequency to Plot, default to 20Hz
    :param lower_cutoff: Minimum Magnitude to Plot, default to 1000
    :param upper_cutoff: Maximum Magnitude to plot, default to 20000
    :param figsize: Size of the figure, default to (12, 6)
    :param xlim: Tuple specifying the limits of the x-axis, default to None
    :param ylim: Tuple specifying the limits of the y-axis, default to None
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
    # Get the one-sided spectrum
    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    #Plot frequency decomposition
    plt.figure(figsize=figsize)
    print(f'The maximum frequency is {(max( (v, i) for i, v in enumerate(new_array[lower_freq_index:upper_freq_index]) )[1])/T}')
    plt.plot(f_oneside[lower_freq_index:upper_freq_index], new_array[lower_freq_index:upper_freq_index])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()

#Averages out complex number every 1Hz in Frequency Domain
def average_out_harmonics(signal,f0=1,fmax=100):
    """
    :param signal: signal to average out 1Hz harmonics for
    :param f0: Fundamental frequency
    :param fmax: Maximum harmonic to filter
    :return: Filtered FFT data
    """
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


#Averages out outliers in Frequency Domain
def remove_outliers_freqDomain(signal,wlen):
    """
    :param signal: signal to average outliers for
    :param wlen: Window Length to consider for average
    :return: Filtered FFT data
    """
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

#Scipy Comb filter--Series of Notch filters every f0 Hz
def scipy_comb_filter(signal, sr=100, f0=1, Q=20):
    """
    :param signal: signal to apply comb filter on
    :param sr: Sampling Frequency aka sampling rate
    :param f0: harmonics to filter out
    :param Q: Quality factor of comb filter
    :return: Filtered time series data and FFT Plot
    """
    b, a = scipy.signal.iircomb(f0, Q, ftype = 'notch', fs=sr)
    print(b)
    print(a)
    #returns filtered time series data
    # Frequency response
    freq, h = scipy.signal.freqz(b, a, fs=sr)
    response = abs(h)
    # To avoid divide by zero when graphing
    response[response == 0] = 1e-20
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(freq, 20*np.log10(abs(response)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([-30, 10])
    ax[0].grid(True)
    ax[1].plot(freq, (np.angle(h)*180/np.pi+180)%360 - 180, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid(True)
    plt.show()
    y_filter = scipy.signal.lfilter(b, a, signal)
    freqData, freqBins = FFTData(y_filter,100)
    FFTPlot(freqData, freqBins,sr=100,lower_freq=.3, upper_freq=20, lower_cutoff=0, upper_cutoff=1000000,title='FFT with scipy comb filter')
    return y_filter



