#%%
import pandas as pd
import numpy as np
import os
from scipy.io import savemat
# import tsfel
#%%
x_data = pd.read_csv('./Patient9_Data_Set1.csv')
#%%
sbs_score = pd.read_excel('./Patient_9_SBS_Scores.xlsx', header=2, usecols='A:C')
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
for index, row in sbs_score.iterrows():
    sbs_time = row['dts']
    sbs_time_start = sbs_time - pd.Timedelta(5, 'minutes')
    sbs_time_end = sbs_time + pd.Timedelta(5, 'minutes')
    conditions = (x_data['dts'] > sbs_time_start) & (x_data['dts'] < sbs_time_end)
    x_mag = x_data[conditions]['mag'].to_numpy()
    if x_mag.shape[0] > 0:
        x_.append(x_mag[:59899])
        y.append(row['SBS'])
    

#%%
x_mag_data = np.vstack(x_)
sbs_data = np.array(y)

# %%
x_train_data = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X_train[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X_train.shape[0])])[:, :, None] for k in range(X_train.shape[1])], axis = 2)

# %%