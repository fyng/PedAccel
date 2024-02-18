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
from sklearn.model_selection import GroupKFold
from sklearn import metrics
#%%
def ordinal_labels(y, num_classes = None):
    if not num_classes:
        num_classes = np.max(y) + 1
    range_values = np.arange(num_classes - 1)[None, :]
    print(range_values.shape)
    range_values = np.tile(range_values, [y.shape[0], 1])
    print(range_values.shape)
    ordinal_label = np.where(range_values < y, 1, 0)
    return ordinal_label
# %%
X = loadmat('../DONOTPUSH/data.mat')['X']
y = loadmat('../DONOTPUSH/data.mat')['y']+3
ids = loadmat('../DONOTPUSH/data.mat')['group']
#%%
selected = np.squeeze(y != y.round)
X = X[selected, :]
ids = np.squeeze(ids)[selected]
y = np.squeeze(y)[selected]
#%%
kf = GroupKFold(n_splits=10)

# %%
# tsfel (https://tsfel.readthedocs.io/en/latest/)
cfg_file = tsfel.get_features_by_domain()
del cfg_file['spectral']["LPCC"]
del cfg_file["spectral"]["MFCC"]
del cfg_file["spectral"]["Spectral roll-on"]
fs = 1/60

# %%
x_features = np.concatenate([np.vstack([tsfel.time_series_features_extractor(cfg_file, X[i, k, :], fs = fs, verbose = 0).to_numpy() for i in range(X.shape[0])])[:, :, None] for k in range(X.shape[1]-2)], axis = 2)

# %%
x_features = x_features.reshape(x_features.shape[0], -1)
X = np.concatenate((x_features, X[:, 3, :].mean(axis = 1)[:, None]), axis=1)
#%%
y = ordinal_labels(np.squeeze(y).reshape(-1, 1), 6)
#%%
for i, (train_idx, test_idx) in enumerate(kf.split(X,y, groups=ids)):
    x_train_data = X[train_idx, :]
    y_train_data = y[train_idx, :]
    x_test_data = X[test_idx, :]
    y_test_data = y[test_idx, :]
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
