#Import Necessary Libraries
import General_Functions
import Smoothing_Functions
import Frequency_Domain
import numpy as np
from pygt3x.reader import FileReader
from matplotlib import pyplot as plt
import Actigraph_Metrics
import Correlation_and_Analysis
import pandas as pd
import skdh
import scipy
import random
import matplotlib.pyplot as plt
import numpy as np

# Load csv
with FileReader("No Movement.gt3x") as reader:
        myData = reader.to_pandas()
print(myData.head(5))
General_Functions.create_time_column(myData)
General_Functions.create_absMag_Column(myData)
myData = myData.iloc[:32000]
Frequency_Domain.FFT(myData['VecMag'], 100, lower_freq=.3, upper_freq=20, lower_cutoff=.3, upper_cutoff=100000,title='FFT of X-axis Data')

General_Functions.plot_Data(myData, 'Time(s)', 'VecMag')
plt.rcParams["figure.figsize"] = [20, 3.50]
plt.rcParams["figure.autolayout"] = True
myX, MAD = Actigraph_Metrics.MAD(myData, wlen=50)

SBSx = [myX[int(108100/50)],myX[int(240000/50)],myX[int(1266000/50)],myX[int(1680000/50)],myX[int(1812000/50)],
        myX[int(2184000/50)], myX[int(2400000/50)], myX[int(2736000/50)], myX[int(2772000/50)], myX[int(3120000/50)], myX[int(3432000/50)],
        myX[int(3804000/50)]]

#print(f'This is SBSx {SBSx}')
#Check date time of SBS marker
#print(f'This is SBSx Check {SBSCheck}')

SBSy = np.array([1,0,2,-1,0,1,0,1,2,-1,1,1])

myX, MAD = Actigraph_Metrics.MAD(myData, wlen=50)
accelerationy = np.array(MAD)
accelerationx = np.array(myX)

ax1 = plt.subplot()
#ax1.set_ylim[-1,2]

l1, = ax1.plot(SBSx,SBSy,  marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green", color='red')
ax2 = ax1.twinx()
#ax2.set_ylim[0,5]
l2, = ax2.plot(accelerationx,accelerationy, color='orange')
plt.legend([l1, l2], ["SBS", "MAD"])

plt.show()