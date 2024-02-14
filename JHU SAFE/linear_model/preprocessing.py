#%%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch.optim as optim
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import tsfel
from sklearn.model_selection import KFold
from sklearn import metrics

# %%
X = loadmat('../DONOTPUSH/data.mat')['X']
y = np.round(loadmat('../DONOTPUSH/data.mat')['y'])  

#%%
kf = KFold(n_splits=10, shuffle = True)

# %%
# tsfel (https://tsfel.readthedocs.io/en/latest/)
cfg_file = tsfel.get_features_by_domain()
del cfg_file['spectral']["LPCC"]
del cfg_file["spectral"]["MFCC"]
del cfg_file["spectral"]["Spectral roll-on"]
fs = 1/60

# %%
x_features = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X.shape[0])])[:, :, None] for k in range(X.shape[1]-1)], axis = 2)

# %%
x_features = x_features.reshape(x_features.shape[0], -1)
X = np.concatenate((x_features, X[:, 3, :].mean(axis = 1)[:, None]), axis=1)
#%%
y = np.squeeze(y[:])
#%%
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    x_train_data = X[train_idx, :]
    y_train_data = y[train_idx]
    x_test_data = X[test_idx, :]
    y_test_data = y[test_idx]
    # normalizing input features
    scaler = preprocessing.StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)
    x_test_data = scaler.transform(x_test_data)

    x_total = np.concatenate((x_train_data, x_test_data), axis = 0)
    x_total = x_total[:, ~np.any(np.isnan(x_total), axis = 0)]
    x_train = x_total[:x_train_data.shape[0], :]
    x_test = x_total[x_train_data.shape[0]:, :]

    filename = f"fold{i}.mat"
    savemat(filename, {'X_train':x_train, 'y_train':y_train_data, 'X_test':x_test, 'y_test':y_test_data})
# %%
