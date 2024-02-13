# %%
from scipy.io import loadmat, savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#%%
    
#%%

# %%
X_train = (loadmat('fold1.mat')['X_train'])
y_train = (loadmat('fold1.mat')['y_train']) 
X_test = (loadmat('fold1.mat')['X_test'])  
y_test = (loadmat('fold1.mat')['y_test'])
y_test = y_test.sum(axis = 1)

# %%
# training
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# %%
plt.scatter(y_test, y_predict[:])
plt.xlabel('actual')
plt.ylabel('predict')
# %%
