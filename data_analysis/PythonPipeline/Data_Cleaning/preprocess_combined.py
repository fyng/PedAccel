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






if __name__ == '__main__':
    data_dir = r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\data_analysis\PythonPipeline\PatientData'
    # data_dir = r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\data_analysis\PythonPipeline\PatientData'
    # load_and_segment_data(data_dir)
    load_and_segment_data_mat(data_dir)