#%%
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#%%
plot_dict = {"metrics":[], "label":[], 'fold_id':[]}
y_preds = []
y_trues = []
l = ["Acc", "F1", "Sens", "Spec", "Kappa"]
#%%
for repeat in range(10):
    filename = f"results/cv{repeat}.mat"
    Acc = loadmat(filename)['Acc']
    F1 = loadmat(filename)['F1']
    Sens = loadmat(filename)['Sens']
    Spec = loadmat(filename)['Spec']
    Kappa = loadmat(filename)['Kappa']
    y_pred = loadmat(filename)['y_pred']
    y_true = loadmat(filename)['y_true']

    labels = np.repeat(l, 10).tolist()
    measures = np.squeeze(np.concatenate((Acc, F1, Sens, Spec, Kappa), axis=1)).tolist()
    plot_dict['metrics'] += measures
    plot_dict['label'] += labels
    plot_dict['fold_id'] += [repeat] * len(labels)
    y_preds += np.squeeze(y_pred).tolist()
    y_trues += np.squeeze(y_true).tolist()
#%%
df = pd.DataFrame(plot_dict)
#%%
folder = "results/"
plot = sns.boxplot(df, x = 'label', y = 'metrics', hue='fold_id', legend=False)
fig = plot.get_figure()
fig.savefig(folder + "overall.png")

#%%
folder = "results/"
cnf = metrics.confusion_matrix(y_trues, y_preds)
cnf_pct = cnf / cnf.sum()
plot = sns.heatmap(cnf_pct, annot = True)
fig = plot.get_figure()
fig.savefig(folder + "confusion_matrix_percent.png")

#%%
folder = "results/"
cnf = metrics.confusion_matrix(y_trues, y_preds)
cnf_pct = cnf 
plot = sns.heatmap(cnf_pct, annot = True)
fig = plot.get_figure()
fig.savefig(folder + "confusion_matrix.png")
# %%
