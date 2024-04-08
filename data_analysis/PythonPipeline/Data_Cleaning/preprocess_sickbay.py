import pandas as pd
import numpy as np
from pygt3x.reader import FileReader 
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat

def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # TODO: define a data model and ingest all the metadata we care about at once elegantly
    '''
    Load data from Excel file

    WARNING! This function assumes that 
    1. the header is on the 3rd row. 
    2. the excel contains a column 'SBS'
    This is very fragile. We should consider the regularity of our excel formatting. 
    '''
    df = pd.read_excel(sbs_filepath, header=0)
    col_names = df.columns.values.tolist()
    if 'SBS' not in col_names:
        raise ValueError('SBS column not found in the excel file')
    if verbose:
        print(df.head())
        print(col_names)
    if to_numpy:
        array = df.to_numpy()
        if verbose:
            print(array.shape)
        return array, col_names

    return df, col_names

def load_segment_sickbay(data_dir, window_size=15, lead_time=10):
    '''
    @param MATLAB SickBay Vitals Files, Created by SickBay_Extraction.py
    Outputs MATLAB files concatenated with their respective SBS scores and heart rate values
    '''
    # search for patient directories in the data directory
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            print('Loading sickbay data')
            sickbay_filepath = os.path.join(patient_dir, patient + '_SickBayData.mat')
            if not os.path.isfile(sickbay_filepath):
                raise FileNotFoundError(f'SickBay file not found: {sickbay_filepath}')
            sickbay_data = loadmat(sickbay_filepath)
            time = sickbay_data["time"]
            hr = sickbay_data["heart_rate"]

            print('Loading SBS data')
            sbs_file = os.path.join(patient_dir, patient + '_SBS_Scores.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            epic_data.dropna(subset=['SBS'], inplace=True)  # drop rows with missing SBS scores
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
            # precompute start and end time for each SBS recording
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')

            print('Processing')
            sbs = []
            heart_rates = []
            for i, row in epic_data.iterrows():
                # in_window = (time > pd.Timestamp(row['start_time'])) & (time < pd.Timestamp(row['end_time']))
                # in_window = (pd.Series(time.flatten()) > pd.Timestamp(row['start_time'])) & (pd.Series(time.flatten()) < pd.Timestamp(row['end_time']))
                start_time_np = np.datetime64(row['start_time'])
                end_time_np = np.datetime64(row['end_time'])
                in_window = (time > np.datetime64(row['start_time'])) & (time < np.datetime64(row['end_time']))
                print('Dimensions of in_window:', in_window.shape)
                print('Dimensions of hr:', hr.shape)
                if np.any(in_window):
                    sbs.append(row['SBS'])
                    heart_rate_in_window = hr[in_window.flatten()]
                    print('Dimensions of heart_rate_in_window:', heart_rate_in_window.shape)
                    heart_rates.append(np.mean(heart_rate_in_window))  # Append mean heart rate in the window
                else:
                    print('No matching sickbay data for SBS recording at ', row['dts'])

            print('Save to file')
            sbs = np.array(sbs)
            heart_rates = np.array(heart_rates)
            print(sbs.shape)
            print(heart_rates.shape)

            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN.mat'
            save_file = os.path.join(patient_dir, filename)
            savemat(save_file, {'sbs': sbs, 'heart_rates': heart_rates})

if __name__ == '__main__':       
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)