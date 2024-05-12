#%%
import pandas as pd
import numpy as np
import random
from scipy.io import loadmat, savemat
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim

#%%
filename = "../DataAnalysis/FeatureExtraction/Patient9_data_set1.csv"
# %%
data = pd.read_csv(filename)
X = data.iloc[:100][["X", "Y", "Z"]]
available_sbs_scores = [-3, -2, -1, 0, 1, 2]
y = [random.choice(available_sbs_scores) for _ in range(100)]
# %%
X = X.to_numpy()
y = np.array(y)
#%%
X = torch.from_numpy(X)
y = torch.from_numpy(y)
#%%
y = y.reshape(-1, 1)
#%%
#sigmoid ((100 * 3) x (3 * 2)) x (2, 1)= 100 * 1
# x = (100 * 3)
# h = sigmoid ((100 * 3) x (3 * 2))
# y = h x (2 * 1)= 100 * 1

# (1, 3) -> (1, 2) -> (1, 1)

layer1 = nn.Linear(3, 1, bias=False)
optimizer = optim.Adam(layer1.parameters(), lr=1e-2)
# %%
X = X.to(torch.float32)
y = y.to(torch.float32)

# %%
def training(model, opt):
    for iter in range(10):
        #forward pass
        out = model(X)
        #calculating loss
        loss = torch.square(y - out).sum()
        #backward pass
        loss.backward()
        opt.step()
        with torch.no_grad():
            #metrics ... 
            pass
        print(f"iter {iter}: {loss.detach().item()}")
# %%
training()
# %%
class model(nn.Module):
    def __init__(self, in_features, out_features, num_hidden):
        super(model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, num_hidden, bias=False),
            nn.Sigmoid(),
            nn.Linear(num_hidden, out_features, bias=False)
        )
        self.bias = nn.Parameter(torch.rand(1))
    def forward(self, X: torch.tensor):
        return self.model(X) + self.bias

#%%
neural_network = model(3, 1, 2)
# %%
neural_network_opt = optim.Adam(neural_network.parameters(), lr=1e-3)

# %%
training(neural_network, neural_network_opt)
# %%
