#%%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import torch.optim as optim
from sklearn import preprocessing
import itertools
import tsfel


#%%
class ordinal_regression(nn.Module):
    def __init__(self, ndim, ydim):
        super(ordinal_regression, self).__init__()
        self.w = self.w = nn.Linear(ndim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, ydim)))
   
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_hat = self.w(X)
        y_hat = F.sigmoid(y_hat.repeat(1,4) + self.bias)
        return y_hat
   
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_hat = self.w(X)
            y_hat = F.sigmoid(y_hat.repeat(1,4) + self.bias)
            return y_hat.numpy()
        
#%%
def training(model, X, y,max_iter):
    learning_rate = 0.0001
    epsilon = 1e-7
    criterion = torch.nn.BCELoss()
    opt_j     = optim.Adam(model.parameters(), lr=learning_rate)
    y_train = torch.Tensor(y)
    X_train = X
    loss_criterion = []
    loss_sparsity = []
    old_loss = 0
    for i in range(max_iter):
        opt_j.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss_c = criterion(outputs, y_train) 
        loss_criterion.append(loss_c.detach().item())
        loss = loss_c
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        opt_j.step() # Updates weights and biases with the optimizer (SGD)
        if abs(old_loss - loss) < epsilon:
            break
        old_loss = loss.detach().item()
        
    return loss_criterion, loss_sparsity, model.w.weight.detach().numpy()

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
y_train_data = ordinal_labels(y_train.reshape(-1, 1), 6)
y_test_data = ordinal_labels(y_test.reshape(-1, 1), 6)
# %%
x_train_data = x_train_data.reshape(x_train_data.shape[0], -1)
x_test_data = x_test_data.reshape(x_test_data.shape[0], -1)
# %%
# normalizing input features
scaler = preprocessing.StandardScaler()
x_train_data = scaler.fit_transform(x_train_data)
x_test_data = scaler.transform(x_test_data)
# %%
filename = "fold1.mat"
savemat(filename, {'X_train':x_train_data, 'y_train':y_train_data, 'X_test':x_test_data, 'y_test':y_test_data})
# %%
