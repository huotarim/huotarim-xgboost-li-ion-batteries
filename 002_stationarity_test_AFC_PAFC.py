# Time series model selection and multi-step forecating for battery data
import pandas as pd
from pandas import to_datetime
from pandas import DataFrame
from pandas import read_csv
import numpy as np
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
from math import sqrt
from math import log
from sklearn.metrics import mean_squared_error

# split a univariate dataset into train1/ test1 sets containing both X and y (last value on a row)
def train_test_split_all(data, n_testratio):
    n_test = int(len(data) * n_testratio)
    return data[:-n_test], data[-n_test:]

# statistical test for the stationarity of the time series
from pandas import Series
# create a differenced (previous step) time series
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return Series(diff)

# main
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'JPmth023.csv'
filename = DIR + FILE

data = read_csv(filename, usecols = ['TimeDate','SOH'], parse_dates= True, index_col = 'TimeDate',squeeze=True)
# split all data to train and test; use train only for walk-forward validation
test_ratio= 0.4
data_train, data_test = train_test_split_all(data, test_ratio)

# check if stationary
# (1) non-differenced data
y = data_train.values
y = Series(y)
y.index = data_train.index[0:]
print("y non-diff\n", y)

result = adfuller(y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ACF and PACF plots of (non-transformed) time series
# Upper AFC
fig = pyplot.figure(figsize=(7,10))
ax = fig.add_subplot(211)
plot_acf(data_train, lags=12, ax=pyplot.gca(), title = 'Battery (a) autocorrelation')
pyplot.title('Battery (a) with non-differenced data', y= 1.1, loc = 'right')
pyplot.xlabel('Lag (months)')
pyplot.ylabel('Correlation (-1..1)')
# Lower PAFC
ax2 = fig.add_subplot(212)
plot_pacf(data_train, lags=12, ax=pyplot.gca(), title = 'Battery (a) partial autocorrelation')
pyplot.xlabel('Lag (months)')
pyplot.ylabel('Correlation (-1..1)')
pyplot.savefig('Fig_a_AFCPAFC_0.pdf')
pyplot.show()

# (2) transform data
y = difference(y)
y.index = data_train.index[1:]
print("y diff\n", y)

# check if stationary
result = adfuller(y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ACF and PACF plots of (non-transformed) time series
# Upper AFC
fig = pyplot.figure(figsize=(7,10))
ax = fig.add_subplot(211)
plot_acf(data_train, lags=12, ax=pyplot.gca(), title = 'Battery (a) autocorrelation')
pyplot.title('Battery (a) with 1 month differencing', y= 1.1, loc = 'right')
pyplot.xlabel('Lag (months)')
pyplot.ylabel('Correlation (-1..1)')
# Lower PAFC
ax2 = fig.add_subplot(212)
plot_pacf(data_train, lags=12, ax=pyplot.gca(), title = 'Battery (a) partial autocorrelation')
pyplot.xlabel('Lag (months)')
pyplot.ylabel('Correlation (-1..1)')
pyplot.savefig('Fig_a_AFCPAFC_1.pdf')
pyplot.show()

