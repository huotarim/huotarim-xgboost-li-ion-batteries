# Make Wilcoxon for pair of batteries; make simple data transforamtion 
# as in ARIMA model to smoothen out different time sereis change and calibration issues.
#
# wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import rand
from scipy.stats import wilcoxon

# load data from different directories
import pandas as pd
from pandas import datetime
from pandas import read_csv
import numpy as np
import os

DATA_DIR = "../Logsn/ind_and_selBcol/v140/JPmth/"

def stats(Bseries):
    # compare samples
    minC = Bseries[['TemperatureEnvironment_C']].min()
    maxC = Bseries[['TemperatureEnvironment_C']].max()
    meanC = Bseries[['TemperatureEnvironment_C']].mean()
    cyc = Bseries['cyc'].iloc[-1]
    SOH = Bseries[['SOH']].mean()
    return minC, maxC, meanC, cyc, SOH

# Differencing the time series
from pandas import Series
# create a differenced time series between two time steps
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return Series(diff)

# calculate H0/ H1 Wilcoxon
H0 = 0
H1 = 0
for _battery_fn in os.listdir(DATA_DIR):
    FILE = DATA_DIR + _battery_fn
    FILE2 = DATA_DIR + "JPmth023.csv"
    if _battery_fn == "JPmth023.csv" or _battery_fn == ".DS_Store" or _battery_fn == "JN*":
        continue #start the loop from the beginning
    print("trying : ", FILE)

    series = read_csv(FILE, usecols = ['TimeDate','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 
    series2 = read_csv(FILE2, usecols = ['TimeDate','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 

    # Transform data
    X = series.values
    X = difference(X)
    X.index = series.index[1:]
    X2 = series2.values
    X2 = difference(X2)
    X2.index = series2.index[1:]
    
    Bseries = read_csv(FILE, usecols = ['TimeDate', 'cyc', 'TemperatureEnvironment_C','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 
    Bseries2 = read_csv(FILE2, usecols = ['TimeDate', 'cyc', 'TemperatureEnvironment_C','SOH'],parse_dates= True, index_col = 'TimeDate',squeeze=True) 

    if (len(series) != len(series2)):
        print('> Not same length (sample battery/ battery 23) %d, %d' % (len(series),len(series2)))
        continue
    else:
        print('%s vs. %s' % (_battery_fn, "JPmth023.csv"))

    # compare samples
    stat, p = wilcoxon(X, X2, correction=True)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
        minC, maxC, meanC, cyc, SOH = stats(Bseries)
        print('Ambient temperature %.1f-%.1f C; mean %.1f' %(minC, maxC ,meanC))
        print('%.1f cycles and %.1f mean SoH' %(cyc, SOH))
        print('p-value: %.3f\n' % p)
        H0 +=1 
    else:
        print('Different distribution (reject H0)')
        minC, maxC, meanC, cyc, SOH = stats(Bseries)
        print('Ambient temperature %.1f-%.1f C; mean %.1f' %(minC, maxC , meanC))
        print('%.1f cycles and %.1f mean SoH\n' %(cyc, SOH))
        H1 += 1

minc, maxC, meanC, cyc, SOH = stats(Bseries2)
print(FILE2)
print('Ambient temperature %.1f-%.1f C; mean %.1f' %(minC, maxC ,meanC))
print('%.1f cycles and %.1f mean SoH\n' %(cyc, SOH))
H0 +=1 
print('Same/ different %d/%d' % (H0,H1))    


