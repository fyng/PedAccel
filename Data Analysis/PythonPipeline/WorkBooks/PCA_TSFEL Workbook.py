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
#use pip install --target=c:'path' package name to install to specific folder

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

# Load Data
os.chdir(r'C:\Users\jakes\Documents\DT 6 Analysis\PythonCode\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')
#%%
filename = 'pt9_win5_5.mat'
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
    #MAD = Actigraph_Metrics.VecMag_MAD(x_mag[i,:],100,threshold = False)
    signal = x_mag[i,:]
    features = tsfel.time_series_features_extractor(cfg_file, signal, fs=100, verbose=0)
    features_list.append(features)
    sbs_list.append(SBS[0][i])


#%%
# Convert features and SBS scores to DataFrame
features_array = np.array(features_list).reshape(-1, 389)
df_features = pd.DataFrame(features_array)
df_features.columns = ['feature_' + str(col) for col in df_features.columns]

df_sbs = pd.DataFrame({'SBS': sbs_list})

#%%
# Concatenate features and SBS scores
df = pd.concat([df_sbs, df_features], axis=1)

print(df.head(10))

# Normalize the data
x = df[df.columns[1:]].values
print("df with SBS removed for analysis:")
print(df[df.columns[1:]].head(5))

x = StandardScaler().fit_transform(x) # normalizing the features

#mean should be zero, std should be 1
print(np.mean(x),np.std(x))


#Store normalized data
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_df = pd.DataFrame(x,columns=feat_cols)
print(normalised_df.head(5))
print('PCA calculations')

#PCA
pca_actigraphy = PCA(n_components=4)
principalComponents_actigraphy = pca_actigraphy.fit_transform(x)
principal_actigraphy_Df = pd.DataFrame(data = principalComponents_actigraphy
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])

#print(principal_actigraphy_Df.head(20))
print('Explained variation per principal component: {}'.format(pca_actigraphy.explained_variance_ratio_))
print('Plotting PCA')


#Plot PCA
plot = plt.figure(figsize=(12,12))
plt.figure(figsize=(12,12))
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Actigraphy and SBS",fontsize=20)


#Color code points based on SBS
for i in range(len(df['SBS'])):
    if df['SBS'][i] == -1:
        color = 'purple'
    if df['SBS'][i] == 0:
        color = 'blue'
    if df['SBS'][i] == 1:
        color = 'orange'
    if df['SBS'][i] == 2: 
        color = 'red'
    plt.scatter(principal_actigraphy_Df.loc[i, 'principal component 1'], principal_actigraphy_Df.loc[i, 'principal component 2'], c = color, s = 50)
    
#Manually create a legend
neg1 = mlines.Line2D([], [], color='purple', marker='o', ls='', label='SBS -1')
zero = mlines.Line2D([], [], color='blue', marker='o', ls='', label='SBS 0')
one = mlines.Line2D([], [], color='orange', marker='o', ls='', label='SBS 1')
two = mlines.Line2D([], [], color='red', marker='o', ls='', label='SBS 2')
plt.legend(handles=[neg1, zero, one, two])
plt.show()

