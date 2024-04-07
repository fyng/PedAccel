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
    Outputs MATLAB files concatenated with their respective SBS scores
    '''
    # search for patient directories in the data directory
    for patient in os.listdir(data_dir):
        # filter out non-directories
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            print('Processing:', patient)

            print('Loading sickbay data')
            sickbay_filepath = os.path.join(patient_dir, patient + 'SickBayData.mat')
            if not os.path.isfile(sickbay_filepath):
                raise FileNotFoundError(f'SickBay file not found: {sickbay_filepath}')
            sickbay_data = loadmat(sickbay_filepath)
            time = sickbay_data["time"]
            hr = sickbay_data["heart_rate"]
            # acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            # sickbay_data[]
            # acti_data, acti_names = load_gt3x_data(actigraphy_filepath)
            # acti_data['mag'] = np.linalg.norm(acti_data[['X', 'Y', 'Z']].values, axis=1)
            # acti_data['dts'] = pd.to_datetime(acti_data['Timestamp'], unit='s')
            # print(acti_data.shape)
            # print(acti_names)

            print('Loading SBS data')
            sbs_file = os.path.join(patient_dir, patient + '_SBS_Scores.xlsx')
            if not os.path.isfile(sbs_file):
                raise FileNotFoundError(f'Actigraphy file not found: {sbs_file}')
            epic_data, epic_names = load_from_excel(sbs_file)
            epic_data.dropna(subset=['SBS'], inplace = True) # drop rows with missing SBS scores
            print(epic_data.shape)
            print(epic_names)
            epic_data['dts'] = pd.to_datetime(epic_data['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
            # precompute start and end time for each SBS recording
            epic_data['start_time'] = epic_data['dts'] - pd.Timedelta(lead_time, 'minutes')
            epic_data['end_time'] = epic_data['dts'] + pd.Timedelta(window_size - lead_time, 'minutes')

            print('Processing')
            windows = []
            sbs = []
            for i, row in epic_data.iterrows():
                # don't like the for-loop, but its not a big bottleneck for the number of SBS recordings we are getting right now. 

                in_window = acti_data[(acti_data['dts'] > row['start_time']) & (acti_data['dts'] < row['end_time'])].loc[:, ['dts', 'mag']]
                in_window.rename(columns={'mag': f'mag_{i}'}, inplace=True)
                if in_window.shape[0] > 0:
                    sbs.append(row['SBS'])
                    in_window['dts'] = in_window['dts'] - row['start_time']
                    windows.append(in_window)
                else:
                    print('No matching accelerometry data for SBS recording at ', row['dts'])

            print('Save to file')
            windows_merged = reduce(lambda  left,right: pd.merge(left,right,on=['dts'], how='outer'), windows)
            windows_merged.drop('dts', axis=1, inplace=True)
            windows_merged = windows_merged.apply(pd.to_numeric, downcast='float') #float32 is enough
            windows_merged.interpolate(axis=1, inplace=True) #fill na with linear interpolation

            x_mag = np.transpose(windows_merged.to_numpy())
            assert not np.isnan(np.sum(x_mag)) # fast nan check
            sbs = np.array(sbs)
            print(x_mag.shape)
            print(sbs.shape)

            filename = f'{patient}_{lead_time}MIN_{window_size - lead_time}MIN.mat'
            save_file = os.path.join(patient_dir, filename)
            savemat(save_file, dict([('x_mag', x_mag), ('sbs', sbs)]))