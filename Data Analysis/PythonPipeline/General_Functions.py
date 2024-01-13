#Import Necessary Libraries
import colorcet as cc
from colorcet import fire
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf

#Create a time column
def create_time_column(myData):
    """
        This Function Creates a time column in the Accelerometry Data Frame
        :param myData: gt3x file containing accelerometry data
        :return: Updated Data Frame with Time Column
    """
    frequency = 100 #Hertz
    time_Column_Array = []
    #length of time column is the same as the length of a data column
    length_of_Data = len(myData["X"])
    for i in range(1, length_of_Data + 1):
        time_Column_Array.append(i*(1/frequency))
    myData['Time(s)'] = time_Column_Array
    print(myData.head(5))

#Create a Vector Magnitude column
def create_ENMO_Column(myData):
    """
        This Function Creates a Vector Magnitude minus one column
        :param myData: gt3x file containging accelerometry data
        :return: Updated Data Frame with Vector Magnitude Column
    """
    ENMO = (((myData["X"])**2)  +  ((myData["Y"])**2)  + ((myData["Z"])**2))
    ENMO = np.sqrt(ENMO) -1
    myData['ENMO'] = ENMO

def create_absMag_Column(myData):
    VecMag = np.sqrt((((myData["X"]) ** 2) + ((myData["Y"]) ** 2) + ((myData["Z"]) ** 2)))
    myData['VecMag'] = VecMag

#Plot Two columns of Data
def plot_Data(myData,xData_name,yData_name,title='Raw Data'):
    """
           This Function Creates a scatter plot given ultra large data sets
           :param myData: gt3x file containing accelerometry data, xData_name: title of x data column, yData_name: title of y data column
           :return: Plot of X vs Y Data
    """
    x_range = myData[xData_name].min(), myData[xData_name].max()
    y_range = myData[yData_name].min(), myData[yData_name].max()
    print("x_range: {0} y_range: {1}".format(x_range, y_range))
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=300,plot_width=600)  # auto range or provide the `bounds` argument
    agg = cvs.points(myData, xData_name, yData_name)  # this is the histogram
    img = ds.tf.set_background(ds.tf.shade(agg, cmap=cc.fire), "black").to_pil()  # create a rasterized image
    plt.imshow(img)
    plt.xlabel("Time")
    plt.ylabel("Vector Magnitude")
    plt.title(title)
    plt.show()

#Slice Data Function--> Ignore and do not use for now
def Sliced_data(myData, video_one, video_two,start_time_difference, sr=100, cam_start_time_is_larger=True):
    """
           This Function Creates a sliced data frame
           :param myData: gt3x file containing accelerometry data
           :param lower_bound: first video number
           :param upper_bound: last video number
           :return: Updated Data Frame that is sliced from the first video number to the last video number
    """
    if cam_start_time_is_larger:
        lower_bound = ((video_one-1) * 10*60*sr) - (start_time_difference*sr)
        upper_bound = ((video_two-1) * 10 * 60 *sr) - (start_time_difference*sr)
    else:
        lower_bound = ((video_one - 1) * 10 * 60 * sr) + (start_time_difference*sr)
        upper_bound = ((video_two - 1) * 10 * 60 * sr) + (start_time_difference*sr)
    return myData.iloc[lower_bound:upper_bound]

