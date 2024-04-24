# Filtering

import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import math 


def checkVitals(vital_array, window_size, vital_type):
    for index, val in enumerate(vital_array):
        if vital_type == 'hr':
            if val > 220 or val < 30:
                vital_array[index] = np.NaN
        elif vital_type == 'rr':
            if val > 80 or val < 10:
                vital_array[index] = np.NaN
        elif vital_type == 'spo2':
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
    return interpolate_nan(vital_array, window_size)


def interpolate_nan(signal, window_size):
    num_data_points = window_size * 30 # Assumes 0.5Hz data recordings
    nan_count = 0

    for i in signal:
        if math.isnan(i):
            nan_count += 1
    if nan_count >= 5:
        return False    
    
    elif nan_count > 0:
         # Find indices of NaN values
        nan_indices = np.where(np.isnan(signal))[0]  # Accessing the first element of the tuple returned by np.where()
    
        # Fill in missing values using linear interpolation
        not_nan_indices = np.where(~np.isnan(signal))[0]  # Accessing the first element of the tuple returned by np.where()
        signal[nan_indices] = np.interp(nan_indices, not_nan_indices, signal[not_nan_indices])
    
        # Handle extrapolation at the ends if necessary
        if nan_indices[0] == 0:
            signal[:nan_indices[0]] = signal[nan_indices[0]]
        if nan_indices[-1] == len(signal) - 1:
            signal[nan_indices[-1] + 1:] = signal[nan_indices[-1]]

        return True  # Return True to indicate that interpolation/extrapolation was performed
    else: 
        return True; 
    

#test

if __name__ == '__main__':
    data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
   # data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'

    lead_time = 5
    slice_size_min = 10
    sr = .5

    #There is no error handling in place, the .mat file must exist
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            data_filepath_accel = os.path.join(patient_dir, f'{patient}_{lead_time}MIN_{slice_size_min - lead_time}MIN.mat')           
            data_filepath_vitals = os.path.join(patient_dir, f'{patient}_SICKBAY_{lead_time}MIN_{slice_size_min - lead_time}MIN.mat')

        accel_data = loadmat(data_filepath_accel)
        x_mag = accel_data["x_mag"]
        accel_SBS = accel_data["sbs"]
        
        vitals_data = loadmat(data_filepath_vitals)
        temp_hr = vitals_data['heart_rate']
        temp_SpO2 = vitals_data['SpO2']
        temp_rr = vitals_data['respiratory_rate']
        temp_bps = vitals_data['blood_pressure_systolic']
        temp_bpm = vitals_data['blood_pressure_mean']
        temp_bpd = vitals_data['blood_pressure_diastolic']
        vitals_SBS = vitals_data['sbs'].flatten()
        hr = []
        rr = []
        SpO2 = []
        bpm = []
        bps = []
        bpd = []

        print(f'Before: {temp_bpd.shape} for {patient}')

        for j in range(len(vitals_SBS)):

            if checkVitals(temp_hr[j], slice_size_min, 'hr'):
                hr.append(temp_hr[j])
            if checkVitals(temp_rr[j], slice_size_min, 'rr'):
                rr.append(temp_hr[j])
            if checkVitals(temp_SpO2[j], slice_size_min, 'spo2'):
                SpO2.append(temp_hr[j])
            if checkVitals(temp_bps[j], slice_size_min, 'bps'):
                 bpm.append(temp_hr[j])
            if checkVitals(temp_bpm[j], slice_size_min, 'bpm'):
                bps.append(temp_hr[j])
            if checkVitals(temp_bpd[j], slice_size_min, 'bpd'):
                bpd.append(temp_hr[j])

        #Call Functions for Analysis for each patient here!

        print(f'After: {np.array(bpd).shape} for {patient}')
