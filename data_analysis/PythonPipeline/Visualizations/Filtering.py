# Filtering

import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import math 

def interpolate_nan(signal, window_size):
    num_data_points = window_size * 30 # Assumes 0.5Hz data recordings
    nan_count = 0
    for i in signal:
        if math.isnan(i):
            nan_count += 1

    if nan_count >= 5:
        return False
    
    else:
        # Find indices of NaN values
        nan_indices = np.where(np.isnan(signal))[0]
        
        # Fill in missing values using linear interpolation
        not_nan_indices = np.where(~np.isnan(signal))[0]
        signal[nan_indices] = np.interp(nan_indices, not_nan_indices, signal[not_nan_indices])
        
        # Handle extrapolation at the ends if necessary
        if nan_indices[0] == 0:
            signal[:nan_indices[0]] = signal[nan_indices[0]]
        if nan_indices[-1] == len(signal) - 1:
            signal[nan_indices[-1] + 1:] = signal[nan_indices[-1]]      
              
        return True  # Return True to indicate that interpolation/extrapolation was performed
    

def checkVitals(vital_array, window_size, vital_type):
    for index, val in enumerate(vital_array):
        if vital_type == 'HR':
            if val > 220 or val < 30:
                vital_array[index] = np.NaN
        elif vital_type == 'RR':
            if val > 80 or val < 10:
                vital_array[index] = np.NaN
        elif vital_type == 'SpO2':
            if val > 115 or val < 80:
                vital_array[index] = np.NaN
        elif vital_type == 'bps':
            if val > 150 or val < 50:
                vital_array[index] = np.NaN
        elif vital_type == 'bpd':
            if val > 90 or val < 30:
                vital_array[index] = np.NaN
        elif vital_type == 'bpm':
            if val > 150 or val < 50:
                vital_array[index] = np.NaN
        else: 
            print('Error, improper vital type specified')
            return False
    interpolate_nan(vital_array, window_size)
    return True

