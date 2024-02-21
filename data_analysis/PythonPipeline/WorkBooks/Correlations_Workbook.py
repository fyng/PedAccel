#%%
#give PCA access to modules folder
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r"C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\Modules")
print(sys.version) 
import sysconfig; 
#Print where python looks for packages and where packages are downloaded. pip -V is where pip is installed to. 
#pip show 'package name'
print(sysconfig.get_paths()["purelib"])
#use pip install --target=C:\Users\jakes\.virtualenvs\DT6Analysis\Lib\site-packages package name to install to specific folder
from operator import itemgetter
from math import isnan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from scipy.io import loadmat
import os
import Actigraph_Metrics
import tsfel
from scipy.stats import pearsonr

# Load Data
os.chdir(r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')
#%%
filename = 'pt9_5min_twoside.mat'
x_mag = (loadmat(filename)["x_mag"])
SBS = loadmat(filename)["sbs"]

#%%
# Generate configuration file for feature extraction
cfg_file = tsfel.get_features_by_domain()
print(x_mag.shape)

#%%
# Extract features and restructure data
features_list = []
sbs_list = []
for i in range(x_mag.shape[0]):
    signal = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100,threshold = False)
    #signal = np.array(x_mag[i,:])-1
    #signal = x_mag[i,:]
    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=100, verbose=0)
    features_list.append(features)
    sbs_list.append(SBS[0][i])

#%%
# Convert features and SBS scores to DataFrame
#Find number of features
#print((features_list))
features_array = np.array(features_list).reshape(-1, 389)
df_features = pd.DataFrame(features_array)
#list comprehension for column names
columns = [col for col in list(features_list[0])]
df_features.columns = columns

#Pearson Correlation Coefficient
CCoeff = []
for i in columns:
    y = sbs_list
    x = list(df_features[i])
    corr, _ = pearsonr(y, x)
    CCoeff.append(np.abs(corr))
my_dict = dict(zip(list(columns), list(CCoeff)))

# functional
clean_dict = filter(lambda k: not isnan(my_dict[k]), my_dict)
# dict comprehension
clean_dict = {k: my_dict[k] for k in my_dict if not isnan(my_dict[k])}

#Retrieve N features with best correlation coefficient  
# Initialize N
N = 10
 
# N largest values in dictionary
# Using sorted() + itemgetter() + items()
res = dict(sorted(clean_dict.items(), key=itemgetter(1), reverse=True)[:N])

# printing result
print("The top N value pairs are " + str(res))
