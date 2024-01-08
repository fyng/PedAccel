import numpy as np
import matplotlib.pyplot as plt
import scipy
import skdh

def calc_area_under_curve(x,y):
    """
        :param x: x data
        :param y: y data
        :return: Area under the curve (an integration of the data)
        """
    return np.trapz(y, x, dx=.005, axis=-1)


#Comments to be added later
def find_peaks(x):
    peaks, _ = scipy.signal.find_peaks(x, distance=20)
    peaks2, _ = scipy.signal.find_peaks(x, prominence=1)  # BEST!
    peaks3, _ = scipy.signal.find_peaks(x, width=20)
    peaks4, _ = scipy.signal.find_peaks(x,threshold=0.4)  # Required vertical distance to its direct neighbouring samples, pretty useless
   # plt.subplot(2, 2, 1)
    #plt.plot(peaks, x[peaks], "xr");
    #plt.plot(x);
    #plt.legend(['distance'])
    #plt.subplot(2, 2, 2)
    plt.plot(peaks2, x[peaks2], "ob");
    print(f'number of peaks for prominence parameter is {len(peaks2)}')
    plt.plot(x);
    plt.legend(['prominence'])
    #plt.subplot(2, 2, 3)
    #plt.plot(peaks3, x[peaks3], "vg");
    #plt.plot(x);
    #plt.legend(['width'])
    #plt.subplot(2, 2, 4)
    #plt.plot(peaks4, x[peaks4], "xk");
    #plt.plot(x);
    #plt.legend(['threshold'])
    plt.show()

#Comments to be added later
def MAD(sliced_data,wlen = 100):
    shape = (len(sliced_data['X']), 3)
    arr = np.ones(shape)
    arr[0:,0] = sliced_data['X']
    arr[0:, 1] = sliced_data['Y']
    arr[0:, 2] = sliced_data['Z']
    accel = arr
    return skdh.activity.metric_mad(accel, wlen)