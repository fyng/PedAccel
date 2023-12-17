#%%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        