#!/usr/bin/env python
# coding: utf-8

# Vitals Correlations

# In[1]:


# Import Modules

import sys
sys.path.append("..") #give this script access to all modules in parent directory
import os
from pathlib import Path
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math

# Import Statistical Tests and tsfel
from scipy.io import loadmat
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import tsfel
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statistics

# Import Previous Scripts
import Filtering
import Correlations


# Correlation Function using TSFEL

# In[49]:


def pearson_corr_vitals_sbs(signal, sbs, signal_name, lead_time, slice_size_min, patient):
    '''
    @param signal: vitals signal input
    @param sbs: sbs corresponding to vitals signal
    @param signal_name: name of input signal
    '''
    # Assuming tsfel and other necessary imports are already done

    cfg_file = tsfel.get_features_by_domain()
    features_list = []
    sbs_list = []
    fs = .5
    
    # Assuming signal and sbs are lists
    for i in range(len(signal)):
        if np.mean(np.array(signal[i])) != 0: #check for flag list
            sbs_list.append(sbs[i])
            features = tsfel.time_series_features_extractor(cfg_file, signal[i], fs, verbose=0)
            features_list.append(features) #vertically concatenate features data frame
        else: 
            print(f'flag list detected for {signal_name} at sbs index {i} for {patient}, window ignored')

    print(f'Number of extracted features: {len(list(features_list[0]))}')

    #list comprehension for column names
    columns = [col for col in list(features_list[0])]
    #Reshape data frame
    features_array = np.array(features_list).reshape(-1, len(columns)) 
    df_features = pd.DataFrame(features_array)

    df_features.columns = columns

    #Pearson Correlation Coefficient
    CCoeff = []
    for i in columns:
        y = sbs_list
        myX = list(df_features[i])

        #remove NaN values from features list for correlation calculations
        nan_indices = [i for i, x in enumerate(myX) if math.isnan(x)]
        myX = [x for x in myX if not math.isnan(x)]
        cleaned_y = [val for idx, val in enumerate(y) if idx not in nan_indices]

        #check that more than two values remain and calculate correlation
        if len(myX) == len(cleaned_y) and len(myX)>=2:
            corr, _ = pearsonr(cleaned_y, myX)
            CCoeff.append(np.abs(corr)) #append correlation coefficients for every feature to a single list, ordered by column from df_features
        else:
            print(f'not enough data for a metric for {signal_name} for patient {patient}')

    my_dict = dict(zip(list(columns), list(CCoeff)))


    # functional
    clean_dict = filter(lambda k: not math.isnan(my_dict[k]), my_dict) #remove NaN again just in case pearson failed for some metrics
    # dict comprehension
    clean_dict = {k: my_dict[k] for k in my_dict if not math.isnan(my_dict[k])}

    #Retrieve N features with best correlation coefficient  
    # Initialize N
    N = 5
            
    # N largest values in dictionary
    # Using sorted() + itemgetter() + items()
    res = dict(sorted(clean_dict.items(), key=itemgetter(1), reverse=True)[:N])

    # printing result
    print("The top N value pairs are " + str(res))

    #Plot a histogram
    y = list(res.keys())
    x = list(res.values()) #price
    if(len(y) == 0):
        print(f'No correlations calculated for {signal_name} for patient {patient}')
    else:
        if len(x) != 0:
            # Figure Size
            fig, ax = plt.subplots(figsize =(10 ,5))

            # Horizontal Bar Plot
            ax.barh(y, x)
            
            # Remove axes splines
            for s in ['top', 'bottom', 'left', 'right']:
                ax.spines[s].set_visible(False)
            
            # Remove x, y Ticks
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            
            # Add padding between axes and labels
            ax.xaxis.set_tick_params(pad = 5)
            ax.yaxis.set_tick_params(pad = 10)
            
            # Add x, y gridlines
            ax.grid(color ='grey',
                    linestyle ='-.', linewidth = 0.5,
                    alpha = 0.2)

            # Show top values 
            ax.invert_yaxis()

            #set x axis range
            ax.set_xlim([.8*min(x),1.1*max(x)])
            # Add Plot Title
            ax.set_title(f'Correlation between top features and SBS for\n {patient}_{lead_time}MIN_{slice_size_min - lead_time}MIN {signal_name})',
                                loc ='left', )
                    
            # Show Plot

            plt.show()
                    


# In[2]:


# Set Parameters
# data_dir = 'C:/Users/sidha/OneDrive/Sid Stuff/PROJECTS/iMEDS Design Team/Data Analysis/PedAccel/data_analysis/PythonPipeline/PatientData'
data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
slice_size_min = 15
lead_time = 10
sr = .5


# In[51]:


#There is no error handling in place, the .mat file must exist
for patient in os.listdir(data_dir):
    # filter out non-directories
    print(f"Processing{patient}")
    patient_dir = os.path.join(data_dir, patient)
    if os.path.isdir(patient_dir):
       # data_filepath_accel = os.path.join(patient_dir, f'{patient}_{lead_time}MIN_{slice_size_min - lead_time}MIN.mat')           
        data_filepath_vitals = os.path.join(patient_dir, f'{patient}_SICKBAY_{slice_size_min - lead_time}MIN_{lead_time}MIN.mat')
        
       # accel_data = loadmat(data_filepath_accel)
       # x_mag = accel_data["x_mag"]
       # accel_SBS = accel_data["sbs"].flatten()
        
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
        vitals_list = [hr,rr,SpO2,bpm,bps,bpd]
        vitals_names = ['hr','rr','spo2','bpm','bps','bpd']
        temp_vitals = [temp_hr,temp_rr, temp_SpO2,temp_bpm,temp_bps,temp_bpd] 
        
        flag_list = [0] * (int)(sr * 60 * slice_size_min) #generate a list to insert in place of invalid data, 
        #this list serves as a flag for a window to ignore in the box plot function
        
        
        for j in range(len(vitals_list)): #go through every vitals metric
            print(f'original {vitals_names[j]} vitals array shape: {np.array(temp_vitals[j]).shape} ')
            for i in range(len(vitals_SBS)): #go through every SBS score for each vitals metric
                if (Filtering.checkVitals(temp_vitals[j][i], slice_size_min, vitals_names[j])): #check the data in a single window
                    vitals_list[j].append(temp_vitals[j][i]) #append that single window data to the 2D hr,rr,spo2,bpm,bps,bpd arrays if that window's data is valid
                else:
                    vitals_list[j].append(flag_list) #append an array of zeros for window number i for the jth vitals metric if the data is invalid(i.e. too many NaN points)
                    print(f'{vitals_names[j]} SBS index {i} has insufficient data, zeros appended in place') 
            print(f'final {vitals_names[j]} vitals array shape: {np.array(vitals_list[j]).shape}') #should be the number of SBS scores by the number of samples in a window
        
        
        for signal, name in zip(vitals_list, vitals_names): #2D array for each vitals is input to function
            if np.mean(np.array(signal)) != 0: #check if whole 2D array is empty
                pearson_corr_vitals_sbs(signal, vitals_SBS, name, lead_time, slice_size_min, patient)
            else:
                print(f'flag list detected for {patient} for {name}, all data ignored')

