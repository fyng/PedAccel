''''
Contains Functions to Calculate Correlations between Accelerometry, Vitals Data, and SBS
'''

# Import Packages
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from pygt3x.reader import FileReader 

import preprocess_sickbay
import os
import sys

folder_path = 'C:/path/to'
if folder_path not in sys.path:
    sys.path.append(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline')
    
from Data_Cleaning import preprocess

def pearson_corr_accel_vitals(data_dir, lead_time):
    '''
    Calculates the pearson correlation coefficient between vitals and accelerometry waveform data for every patient
    '''

    # Iterate through each patient data folder
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)
            
            print('Loading actigraphy data')
            actigraphy_filepath = os.path.join(patient_dir, f'{patient}_AccelData.gt3x')
            acti_data, acti_names = preprocess.load_gt3x_data(actigraphy_filepath)
            acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            
             # Load heart rate data
            print('Loading heart rate data')
            hr_data_mat = loadmat(os.path.join(patient_dir, f'{patient}_SickBayData.mat'))
            hr_data = pd.DataFrame({
                'timestamp': pd.to_datetime(hr_data_mat['time'][0]),
                'hr': hr_data_mat['heart_rate'][0]  # Adjust indexing as necessary
            })

            # Resample actigraphy data to the same rate as heart rate data
            print('Resampling data')
            freq = '2S'  # Example frequency; adjust as needed based on HR data rate
            acti_resampled = acti_data.resample(freq).mean().interpolate()
            hr_resampled = hr_data.resample(freq).mean().interpolate()
            
             # Ensure same length by trimming to the common timeframe
            common_start = max(acti_resampled.index.min(), hr_resampled.index.min())
            common_end = min(acti_resampled.index.max(), hr_resampled.index.max())
            acti_resampled = acti_resampled.loc[common_start:common_end]
            hr_resampled = hr_resampled.loc[common_start:common_end]

            # Calculate Pearson correlation coefficient
            correlation, _ = pearsonr(acti_resampled['mag'], hr_resampled['hr'])
            print(f'Pearson correlation coefficient for {patient}:', correlation)
            
def pearson_corr_vitals_sbs(data_dir):
    '''
    Calculates the pearson correlation coefficient between SBS and vitals data for each patient.
    '''
    # Iterate through each patient data folder
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)
            
            # Load Vitals/SBS MATLAB File
            vitals_file = os.path.join(patient_dir, f'{patient}_SICKBAY_5MIN_5MIN.mat')
            if not os.path.isfile(vitals_file):
                raise FileNotFoundError(f'MATLAB file not found: {vitals_file}')
            vitals_data = loadmat(vitals_file)
            
            # Get Data
            sbs_scores = np.array(vitals_data['sbs'])
            hr_data = vitals_data['heart_rate']
            
            print('sbs_scores:', sbs_scores.shape, 'hr_data:', hr_data.shape)
            print('sbs_scores type:', type(sbs_scores), 'hr_data type:', type(hr_data))
            
            try:
                corr_coef, p_value = pearsonr(sbs_scores.flatten(), hr_data.flatten())
                print(f"{patient}: Coefficient: {corr_coef}, P-value: {p_value})")
            except Exception as e:
                print(f"Error processing {patient}: {e}")
            
if __name__ == '__main__':
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    # pearson_corr_accel_vitals(data_dir, 10)
    pearson_corr_vitals_sbs(data_dir)
            
        
