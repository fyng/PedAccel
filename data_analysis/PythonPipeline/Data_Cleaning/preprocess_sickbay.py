import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat

def load_from_excel(sbs_filepath, to_numpy=False, verbose=False):
    # Load data from Excel file
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
            vitals_data_df = pd.DataFrame({'dts': vitals_data['dts'], 'heart_rate': vitals_data['heart_rate'], 'SpO2': vitals_data['SpO2'], 'respiratory_rate': vitals_data['respiratory_rate']
                                           , 'blood_pressure_systolic': vitals_data['blood_pressure_systolic'], 'blood_pressure_mean': vitals_data['blood_pressure_mean']
                                           , 'blood_pressure_diastolic': vitals_data['blood_pressure_diastolic']})
        
            windows = []
            sbs = []
            
            for i, row in epic_data.iterrows():
                # Define the time window
                start_time = row['start_time'] - pd.Timedelta(minutes=lead_time)
                end_time = row['end_time'] + pd.Timedelta(minutes=lead_time)

                # Filter heart rate data within the time window
                in_window = vitals_data_df[(vitals_data_df['dts'] >= start_time) & (vitals_data_df['dts'] <= end_time)]
                if not in_window.empty:  # Check if any data values are found in the window
                    sbs.append(row['SBS'])

                    # Calculate the relative time within the window
                    in_window['dts'] = in_window['dts'] - row['start_time']

                    # Rename columns with unique names based on iteration index 'i'
                    rename_dict = {
                        'heart_rate': f'heart_rate_{i}',
                        'SpO2': f'SpO2_{i}',
                        'respiratory_rate': f'respiratory_rate_{i}',
                        'blood_pressure_systolic': f'blood_pressure_systolic_{i}',
                        'blood_pressure_mean': f'blood_pressure_mean_{i}',
                        'blood_pressure_diastolic': f'blood_pressure_diastolic_{i}'
                    }
                    in_window.rename(columns=rename_dict, inplace=True)

                    # Append data to the list
                    windows.append(in_window)

            # Convert sbs to a numpy array
            sbs = np.array(sbs)
            
            # Further processing and saving...
            print('Save to file')
            if windows:
                windows_merged = reduce(lambda left, right: pd.merge(left, right, on=['dts'], how='outer'), windows)
                windows_merged.drop('dts', axis=1, inplace=True)
                windows_merged = windows_merged.apply(pd.to_numeric, downcast='float') # float32 is enough
                windows_merged.interpolate(axis=1, inplace=True) # fill NA with linear interpolation           
                windows_merged.dropna(inplace=True)  # Drop NaN values

                filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size-lead_time}MIN.mat'
                save_file = os.path.join(patient_dir, filename)
                
                savemat(save_file, {
                    'heart_rate': windows_merged.filter(like='heart_rate').values.T,
                    'SpO2': windows_merged.filter(like='SpO2').values.T,
                    'respiratory_rate': windows_merged.filter(like='respiratory_rate').values.T,
                    'blood_pressure_systolic': windows_merged.filter(like='blood_pressure_systolic').values.T,
                    'blood_pressure_mean': windows_merged.filter(like='blood_pressure_mean').values.T,
                    'blood_pressure_diastolic': windows_merged.filter(like='blood_pressure_diastolic').values.T,
                    'sbs': sbs
                })
            else:
                print("No data found for patient:", patient)

if __name__ == '__main__':
    data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)
