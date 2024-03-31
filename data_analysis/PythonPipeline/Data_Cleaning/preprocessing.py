import pandas as pd
import numpy as np
import os
from scipy.io import savemat
from Modules import preprocess

# USAGE
# pass in master directory of all patients
# each patient needs to have its own directory

# Within a patient directory, 
# All patient files must be prefixed by their folder name. For example:
# Patient9
# |_Patient9_DataPt2.gt3x
# |_Patient9__SBS_Scores.xlsx
# Patient11
# |_Patient11_DataPt2.gt3x
# |_Patient11__SBS_Scores.xlsx

# Profit

data_dir = './PatientData/'
preprocess.load_and_segment_data(data_dir)