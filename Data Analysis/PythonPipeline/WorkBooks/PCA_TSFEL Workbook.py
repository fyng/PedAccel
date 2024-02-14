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

#%%
# Load Data
os.chdir(r'C:\Users\sidha\OneDrive\Sid Stuff\PROJECTS\iMEDS Design Team\Data Analysis\PedAccel\Data Analysis\PythonPipeline\PatientData\Patient9')

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
    features = tsfel.time_series_features_extractor(cfg_file, x_mag[i, :], fs=100, verbose=0)
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

df.head(10)

# Normalize the data
x = df.iloc[:, 1:].values  # Exclude the SBS column
x = StandardScaler().fit_transform(x)

#%%
# Perform PCA
pca_actigraphy = PCA(n_components=4)
principalComponents_actigraphy = pca_actigraphy.fit_transform(x)
principal_actigraphy_Df = pd.DataFrame(data=principalComponents_actigraphy,
                                      columns=['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

print('Explained variation per principal component: {}'.format(pca_actigraphy.explained_variance_ratio_))

# Plot PCA
plt.figure(figsize=(12, 12))
plt.xlabel('Principal Component - 2', fontsize=20)
plt.ylabel('Principal Component - 3', fontsize=20)
plt.title("Principal Component Analysis of Actigraphy and SBS", fontsize=20)

for i in range(len(df['SBS'])):
    if df['SBS'][i] == -1:
        color = 'purple'
    elif df['SBS'][i] == 0:
        color = 'blue'
    elif df['SBS'][i] == 1:
        color = 'orange'
    elif df['SBS'][i] == 2:
        color = 'red'
    plt.scatter(principal_actigraphy_Df.loc[i, 'principal component 2'], principal_actigraphy_Df.loc[i, 'principal component 3'], c=color, s=50)

# Manually create a legend
neg1 = mlines.Line2D([], [], color='purple', marker='o', ls='', label='SBS -1')
zero = mlines.Line2D([], [], color='blue', marker='o', ls='', label='SBS 0')
one = mlines.Line2D([], [], color='orange', marker='o', ls='', label='SBS 1')
two = mlines.Line2D([], [], color='red', marker='o', ls='', label='SBS 2')
plt.legend(handles=[neg1, zero, one, two])
plt.show()

# %%
