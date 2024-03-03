#%%
import pandas as pd
import numpy as np
import os
from scipy.io import savemat
# import tsfel
import sys
# os.chdir(r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\Modules')
import sysconfig;
#Where python looks for libraries
print(sysconfig.get_paths()["purelib"])
#%%
x_data = pd.read_csv('./Patient11_Actigraphy.csv')
#x_data = pd.read_csv('./Users/jakes/Documents/DT 6 Analysis/PythonCode/Patient9_Data_Set1')
#%%
sbs_score = pd.read_excel(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient11\Patient11_SBS_Scores.xlsx', header=2, usecols='A:C')
#%%
sbs_score['dts'] = pd.to_datetime(sbs_score['Time_uniform'], format='%m/%d/%Y %H:%M:%S %p')
x_data['dts'] = pd.to_datetime(x_data['time'], format='mixed')
#%%
acc = x_data[['X', 'Y', 'Z']]
mag = np.linalg.norm(acc, axis=1)
x_data['mag'] = mag
#%%
sbs_score = sbs_score.dropna(axis = 0)
#%%
x_ = []
y = []
time = []
for index, row in sbs_score.iterrows():
    sbs_time = row['dts']
    sbs_time_start = sbs_time - pd.Timedelta(10, 'minutes')
    sbs_time_end = sbs_time + pd.Timedelta(0, 'minutes')
    conditions = (x_data['dts'] > sbs_time_start) & (x_data['dts'] < sbs_time_end)
    x_mag = x_data[conditions]['mag'].to_numpy()
    if x_mag.shape[0] > 0:
        x_.append(x_mag[:59899]) #min length of all x_mag
        y.append(row['SBS'])
        time.append(sbs_time)
#%%
x_mag_data = np.vstack(x_)
sbs_data = np.array(y)
time_data = np.array(time)
# %%
os.chdir(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData\Patient11')
# os.chdir(r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')
savemat('Patient11_10MIN_SW_Time_AllSBS.mat', dict([('x_mag', x_mag_data), ('sbs', sbs_data), ('time', time_data)]))
# %%