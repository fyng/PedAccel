import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from functools import reduce
from scipy.io import loadmat
import math 

heart_rate = []
SpO2 = []
respiratory_rate = []
blood_pressure_systolic = []
blood_pressure_mean = []
blood_pressure_diastolic = []

vitals_list = [heart_rate, SpO2, respiratory_rate, blood_pressure_systolic, blood_pressure_mean,blood_pressure_diastolic]
names = ['heart_rate', 'SpO2', 'respiratory_rate', 'blood_pressure_systolic', 'blood_pressure_mean', 'blood_pressure_diastolic']

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
            print(vitals_data_df.head(10))
            sbs = []
            
            for i, row in epic_data.iterrows():
                # Define the time window
                start_time = row['start_time'] - pd.Timedelta(minutes=lead_time)
                end_time = row['end_time'] + pd.Timedelta(minutes=lead_time)

                # Filter heart rate data within the time window
                in_window = vitals_data_df[(vitals_data_df['dts'] >= start_time) & (vitals_data_df['dts'] <= end_time)]
                print(in_window.head(5))

                if not in_window.empty:  # Check if any data values are found in the window
                    sbs.append(row['SBS'])

                    # Calculate the relative time within the window
                    in_window['dts'] = in_window['dts'] - row['start_time']

                    index = 0
                    for vital in vitals_list:
                        column = names[index]
                        temp_list = in_window[column].tolist()
                        temp_list = [x for x in temp_list if not math.isnan(x)]
                        vital.append(temp_list)
                        index+=1

            print(f'RR: {respiratory_rate}\n')
            print(f'HR: {heart_rate}\n')

            # Convert sbs to a numpy array
            sbs = np.array(sbs)
            
            # Further processing and saving...
            print('Save to file')
            if all(vitals_list):
                    filename = f'{patient}_SICKBAY_{lead_time}MIN_{window_size-lead_time}MIN.mat'
                    save_file = os.path.join(patient_dir, filename)
                    savemat(save_file, {'heart_rate': (heart_rate), 'SpO2': (SpO2),'respiratory_rate': (respiratory_rate),
                        'blood_pressure_systolic': (blood_pressure_systolic),'blood_pressure_mean': (blood_pressure_mean),
                        'blood_pressure_diastolic': (blood_pressure_diastolic),'sbs': sbs})
            else:
                    print("No data found for patient:", patient)


if __name__ == '__main__':
    data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    load_segment_sickbay(data_dir)
