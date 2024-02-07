#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tsfel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from scipy.io import loadmat
import os

# Load Data
os.chdir(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')
#%%
filename = 'pt9_win5_5.mat'
x_mag = (loadmat(filename)["x_mag"])
SBS = loadmat(filename)["sbs"]

# Generate configuration file for feature extraction
cfg_file = tsfel.get_features_by_domain()
x_mag.shape
#%%
# Extract features
# temp = tsfel.time_series_features_extractor(cfg_file, x_mag[0, :], fs=100)
# features = tsfel.time_series_features_extractor(cfg_file, x_mag)
extracted_features = []
for i in range(x_mag.shape[0]):
    extracted_features.append(tsfel.time_series_features_extractor(cfg_file, x_mag[0, :], fs=100, verbose = 0))

#%%
# Convert features to DataFrame
df = pd.DataFrame(extracted_features)
print(df.head())

#%%
# Normalize the data
x = df.values
x = StandardScaler().fit_transform(x)
#%%
# PCA
pca_actigraphy = PCA(n_components=4)
principalComponents_actigraphy = pca_actigraphy.fit_transform(x)
principal_actigraphy_Df = pd.DataFrame(data=principalComponents_actigraphy,
                                       columns=['principal component 1', 'principal component 2',
                                                'principal component 3', 'principal component 4'])
print(principal_actigraphy_Df.head(20))
print('Explained variation per principal component: {}'.format(pca_actigraphy.explained_variance_ratio_))

# Plot PCA
plot = plt.figure(figsize=(12, 12))
plt.figure(figsize=(12, 12))
plt.xlabel('Principal Component - 2', fontsize=20)
plt.ylabel('Principal Component - 3', fontsize=20)
plt.title("Principal Component Analysis of Actigraphy and SBS", fontsize=20)
for i in range(len(SBS[0])):
    if SBS[0][i] == -1:
        color = 'purple'
    elif SBS[0][i] == 0:
        color = 'blue'
    elif SBS[0][i] == 1:
        color = 'orange'
    elif SBS[0][i] == 2:
        color = 'red'
    plt.scatter(principal_actigraphy_Df.loc[i, 'principal component 2'],
                principal_actigraphy_Df.loc[i, 'principal component 3'], c=color, s=50)

# Manually create a legend
neg1 = mlines.Line2D([], [], color='purple', marker='o', ls='', label='SBS -1')
zero = mlines.Line2D([], [], color='blue', marker='o', ls='', label='SBS 0')
one = mlines.Line2D([], [], color='orange', marker='o', ls='', label='SBS 1')
two = mlines.Line2D([], [], color='red', marker='o', ls='', label='SBS 2')
plt.legend(handles=[neg1, zero, one, two])
plt.show(plot)
# %%
