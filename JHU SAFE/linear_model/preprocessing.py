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
# %%
X_train = loadmat('../DONOTPUSH/data.mat')['X_train']
y_train = np.round(loadmat('../DONOTPUSH/data.mat')['y_train'] + 3)  
X_test = loadmat('../DONOTPUSH/data.mat')['X_test']   
y_test = np.round(loadmat('../DONOTPUSH/data.mat')['y_test'] + 3)
# %%
sns.histplot(y_test.ravel())
sns.histplot(y_train.ravel())
# %%
# tsfel (https://tsfel.readthedocs.io/en/latest/)
cfg_file = tsfel.get_features_by_domain()
fs = 1/60

# %%
x_train_data = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X_train[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X_train.shape[0])])[:, :, None] for k in range(X_train.shape[1])], axis = 2)
x_test_data = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X_test[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X_test.shape[0])])[:, :, None] for k in range(X_test.shape[1])], axis = 2)

# %%
x_train_data = x_train_data.reshape(x_train_data.shape[0], -1)
x_test_data = x_test_data.reshape(x_test_data.shape[0], -1)
# %%
# normalizing input features
scaler = preprocessing.StandardScaler()
x_train_data = scaler.fit_transform(x_train_data)
x_test_data = scaler.transform(x_test_data)
# %%
x_total = np.concatenate((x_train_data, x_test_data), axis = 0)
x_total = x_total[:, ~np.any(np.isnan(x_total), axis = 0)]
# %%
x_train = x_total[:x_train_data.shape[0], :]
x_test = x_total[x_train_data.shape[0]:, :]
#%%
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1) 
# %%
filename = "fold1.mat"
savemat(filename, {'X_train':x_train, 'y_train':y_train_data, 'X_test':x_test, 'y_test':y_test_data})
# %%
