# Extract and plot ambient temperatures for the batteries
#
# load data from different directories
import pandas as pd
from pandas import datetime
from pandas import read_csv
import numpy as np
import os

DATA_DIR = "../Logsn/ind_and_selBcol/v140/JPmth/"

# calculate min, max and mean ambient temperature
minC = 0
maxC = 0
meanC = 0
for _battery_fn in os.listdir(DATA_DIR):
    FILE = DATA_DIR + _battery_fn
    FILE2 = DATA_DIR + "JPmth023.csv"
    if  _battery_fn == ".DS_Store" or _battery_fn == "JPmth023.csv":
        continue #start the loop from the beginning

    series = read_csv(FILE, usecols = ['TimeDate', 'cyc', 'TemperatureEnvironment_C','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 
    series2 = read_csv(FILE2, usecols = ['TimeDate', 'cyc', 'TemperatureEnvironment_C','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 

    if (len(series) != len(series2)):
        #print('> Not same length (sample battery/ battery 23) %d, %d' % (len(series),len(series2)))
        continue
    else:
        print('%s vs. %s' % (_battery_fn, "JPmth023.csv"))
    # compare samples
    minC = series[['TemperatureEnvironment_C']].min()
    maxC = series[['TemperatureEnvironment_C']].max()
    meanC = series[['TemperatureEnvironment_C']].mean()
    print('Ambient temperature %.1f..%.1f C; mean %.1f' %(minC, maxC , meanC))
    cyc = series['cyc'].iloc[-1]
    SOH = series[['SOH']].mean()
    print('%.1f cycles and %.1f mean SoH \n' %(cyc, SOH))
    # interpret
