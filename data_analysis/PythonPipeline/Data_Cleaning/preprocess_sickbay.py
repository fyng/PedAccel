import pandas as pd
import numpy as np
from pygt3x.reader import FileReader 
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat

def check_window(vitals_list, vitals_window, sbs, vital, row):
    if not vitals_window.empty:  # Check if any heart rate values are found in the window
        sbs.append(row['SBS'])
                    
        # Calculate the relative time within the window
        vitals_window['dts'] = vitals_window['dts'] - row['start_time']
                    
        # Append heart rate values and corresponding relative time to the list
        vitals_list.append(vitals_window[vital].tolist())
        print(len(vitals_window[vital]))
        sbs = np.array(sbs)
    else:
        print(f'No matching {vital} data for SBS recording at ', row['dts'])

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

def load_segment_sickbay(data_dir, window_size=10, lead_time=5):
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
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='mixed')
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')
            
            # Load heart rate data
            vitals_file = os.path.join(patient_dir, f'{patient}_SickBayData.mat')
            if not os.path.isfile(vitals_file):
                raise FileNotFoundError(f'Heart rate file not found: {vitals_file}')
            vitals_data = loadmat(vitals_file)
            time_data = vitals_data['time'][0].flatten()  # Flatten nested array
            time_strings = [item[0] for item in time_data]  # Extract datetime strings

            # Convert datetime strings to datetime objects
            vitals_data['dts'] = pd.to_datetime([str(item) for item in time_strings], format='mixed')
            vitals_data['heart_rate'] = vitals_data['heart_rate'].flatten()  # Flatten heart rate array
            vitals_data['SpO2'] = vitals_data['SpO2'].flatten()  # Flatten heart rate array
            vitals_data['respiratory_rate'] = vitals_data['respiratory_rate'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_systolic'] = vitals_data['blood_pressure_systolic'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_mean'] = vitals_data['blood_pressure_mean'].flatten()  # Flatten heart rate array
            vitals_data['blood_pressure_diastolic'] = vitals_data['blood_pressure_diastolic'].flatten()  # Flatten heart rate array


            # Create a DataFrame from the dictionary
            heart_rate_df = pd.DataFrame({'dts': vitals_data['dts'], 'heart_rate': vitals_data['heart_rate']})
            SpO2_df = pd.DataFrame({'dts': vitals_data['dts'], 'SpO2': vitals_data['SpO2']})
            respiratory_rate_df = pd.DataFrame({'dts': vitals_data['dts'], 'respiratory_rate': vitals_data['respiratory_rate']})
            blood_pressure_systolic_df = pd.DataFrame({'dts': vitals_data['dts'], 'blood_pressure_systolic': vitals_data['blood_pressure_systolic']})
            blood_pressure_mean_df = pd.DataFrame({'dts': vitals_data['dts'], 'blood_pressure_mean': vitals_data['blood_pressure_mean']})
            blood_pressure_diastolic_df = pd.DataFrame({'dts': vitals_data['dts'], 'blood_pressure_diastolic': vitals_data['blood_pressure_diastolic']})

            hr_sbs = []
            SpO2_sbs = []
            rr_sbs = []
            bps_sbs = []
            bpm_sbs = []
            bpd_sbs = []
            heart_rates = []
            SpO2 = []
            respiratory_rate = []
            blood_pressure_systolic = []
            blood_pressure_mean = []
            blood_pressure_diastolic = []
            
            for i, row in epic_data.iterrows():
                # Define the time window
                start_time = row['start_time'] - pd.Timedelta(minutes=10)
                end_time = row['end_time'] + pd.Timedelta(minutes=5)

                # Filter heart rate data within the time window
                hr_in_window = heart_rate_df[(heart_rate_df['dts'] > start_time) & (heart_rate_df['dts'] < end_time)]
                SpO2_in_window = SpO2_df[(SpO2_df['dts'] > start_time) & (SpO2_df['dts'] < end_time)]
                rr_in_window = respiratory_rate_df[(heart_rate_df['dts'] > start_time) & (respiratory_rate_df['dts'] < end_time)]
                bps_in_window = blood_pressure_systolic_df[(blood_pressure_systolic_df['dts'] > start_time) & (blood_pressure_systolic_df['dts'] < end_time)]
                bpm_in_window = blood_pressure_mean_df[(blood_pressure_mean_df['dts'] > start_time) & (blood_pressure_mean_df['dts'] < end_time)]                
                bpd_in_window = blood_pressure_diastolic_df[(blood_pressure_diastolic_df['dts'] > start_time) & (blood_pressure_diastolic_df['dts'] < end_time)]

                check_window(heart_rates, hr_in_window, hr_sbs, 'heart_rate', row)
                check_window(SpO2, SpO2_in_window, SpO2_sbs, 'SpO2', row)
                check_window(respiratory_rate, rr_in_window, rr_sbs, 'respiratory_rate', row)
                check_window(blood_pressure_systolic, bps_in_window, bps_sbs, 'blood_pressure_systolic', row)
                check_window(blood_pressure_mean, bpm_in_window, bpm_sbs, 'blood_pressure_mean', row)
                check_window(blood_pressure_diastolic, bpd_in_window, bpd_sbs, 'blood_pressure_diastolic', row)

        
            # Further processing and saving...
            
            filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size - lead_time}MIN.mat'
            save_file = os.path.join(patient_dir, filename)
            savemat(save_file, {'heart_rate_sbs': hr_sbs, 'heart_rate': heart_rates, 'SpO2_sbs': SpO2_sbs, 'SpO2': SpO2,'respiratory_rate_sbs': rr_sbs, 
                                'respiratory_rate': respiratory_rate, 'blood_pressure_systolic_sbs': bps_sbs, 'blood_pressure_systolic': 
                                blood_pressure_systolic, 'blood_pressure_mean_sbs': bpm_sbs,'blood_pressure_mean': blood_pressure_mean, 
                                'blood_pressure_diastolic_sbs': bpd_sbs,'blood_pressure_diastolic': blood_pressure_diastolic})

if __name__ == '__main__':       
    data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    #data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)