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
    # Iterate through patient directories
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            # Load SBS data
            sbs_file = os.path.join(patient_dir, f'{patient}_SBS_Scores.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'EPIC file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            epic_data.dropna(subset=['SBS'], inplace=True)
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='%m/%d/%Y %I:%M:%S %p')
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')

            # Load heart rate data
            hr_file = os.path.join(patient_dir, f'{patient}_SickBayData.mat')
            if not os.path.isfile(hr_file):
                raise FileNotFoundError(f'Heart rate file not found: {hr_file}')
            heart_rate_data = loadmat(hr_file)
            time_data = heart_rate_data['time'][0].flatten()  # Flatten nested array
            time_strings = [item[0] for item in time_data]  # Extract datetime strings

            # Convert datetime strings to datetime objects
            heart_rate_data['dts'] = pd.to_datetime([str(item) for item in time_strings], format='%m/%d/%Y %I:%M:%S %p')
            print(heart_rate_data['dts'])
                    
            heart_rate_values = heart_rate_data['heart_rate'].flatten()  # Assuming 'HeartRate' is directly accessible in the dictionary
            print(heart_rate_values)
            print(len(heart_rate_values))
            sbs = []
            heart_rates = []
            for _, row in epic_data.iterrows():
                in_window = (heart_rate_data['dts'] > row['start_time']) & (heart_rate_data['dts'] < row['end_time'])
                if np.any(in_window):
                    sbs.append(row['SBS'])
                    heart_rate_in_window = heart_rate_values[in_window]  # Filter 'HeartRate' based on the condition
                    heart_rates.append(heart_rate_in_window)  # Append all heart rate values in the window
                else:
                    print('No matching heart rate data for SBS recording at ', row['dts'])

            # Save to file
            sbs = np.array(sbs)
            heart_rates = np.array(heart_rates)

            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN.mat'
            save_file = os.path.join(patient_dir, filename)
            savemat(save_file, {'sbs': sbs, 'heart_rates': heart_rates})

if __name__ == '__main__':       
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)