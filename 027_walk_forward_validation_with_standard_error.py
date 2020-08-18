# Implementation of walk-forward validation for XGBoost model development
#
# battery 023, walk-forward validation
from pandas import to_datetime
from xgboost import XGBRegressor
from numpy import array
from pandas import DataFrame
from pandas import read_csv
from numpy import mean
from numpy import std
from scipy.stats import sem
#from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

# split a univariate dataset into train1/ test1 sets
def train_test_split_all(data, n_testratio):
    n_test = int(len(data) * n_testratio)
    return data[:-n_test], data[-n_test:]

# split a set (=original train1) to train_X, train_y, test_X, test_y for walk-forward
def train_validation_split(data, timesteps, n):
    # input sequence (t-n, ... t-1); forecast sequence t
    n_train_start = n 
    n_train_stop = n + timesteps
    train_X = array(data[n_train_start : n_train_stop, 0 : -2])
    train_y = array(data[n_train_start : n_train_stop, -1])
    test_X = data[n_train_stop + 1, 0 : -2]
    test_y = data[n_train_stop + 1, -1]
    return train_X, test_X, train_y, test_y

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt((actual - predicted) * (actual - predicted))

# fit a model
def model_fit(train_X, train_y, config):
    # unpack config
    n_estimator, max_depth, learning_rate, colsample_bylevel, n_jobs = config
    # define model
    model = XGBRegressor(n_estimator=n_estimator, max_depth=max_depth, 
    learning_rate=learning_rate, colsample_bylevel=colsample_bylevel, n_jobs=n_jobs)
    # fit model
    model.fit(train_X, train_y)
    return model

# forecast with a pre-fit model
def model_predict(model, history):
    yhat = model.predict(history)
    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, timesteps, data_head_index, cfg):
    # split dataset; in walk-foirward validation test is typically 1.
    train_X, test_X, train_y, test_y = train_validation_split(data, timesteps, data_head_index)
    # print("train_X %s \n test_X %s \n head_index: %i \n" % (train_X, test_X, data_head_index) )
    # fit model
    model = model_fit(train_X, train_y, cfg)
    # seed history with training dataset 
    history = train_X
    #print("lenght %i of test %s, \n length of %i of history %s" % (len(test), test, len(history), history))
    # fit model and make forecast for history
    yhat = model_predict(model, history)
    # estimate prediction error
    error = measure_rmse(yhat, test_y)
    print(' > %.3f' % error)
    return error

# repeat evaluation of a config; default 5 times (as of now dataset has only few = 32 timesteps...)
def repeat_evaluate(data, config, timesteps, n_repeats=5):
    # fit and evaluate the model n times indicated by n_repeats; i is offset to drop history from the head 
    scores = [walk_forward_validation(data, timesteps, i, config) for i in range(n_repeats)]
    return scores

# summarize model performance
def summarize_scores(name, scores):
    # print a summary
    scores_m, score_std = mean(scores), sem(scores)
    print('%s: RMSE mean=%.3f se=%.3f' % (name, scores_m, score_std))
    # box and whisker plot
    pyplot.boxplot(scores)
    pyplot.show()

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'JPmth023.csv'
filename = DIR + FILE

# Attach labels to columns
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']

# Below if swapping to file from one month interval to one minute interval data.
# Reuslts differ only little.
raw_data = read_csv(filename, header=0)
raw_data['date']=to_datetime(raw_data['TimeDate'])
dataset = raw_data.loc[:,names]
dataset = dataset.set_index(raw_data.date)
dataset = dataset.resample('M').mean()

# Split-out validation dataset to test and validation sets; last is SOH
data = dataset.values
#print("data.shape", data.shape)

# Split all data to train and test; use train only for walk-forward validation
test_ratio= 0.4
data_train, data_test = train_test_split_all(data, test_ratio)
#print("data_train.shape", data_train.shape)
#print("data_train.shape[-1:, 0:10] %s [:-1, 11] %s" % (data_train[:-1, 0:-2],data_train[:-1, -1]))

# define config 
# (n_estimator=200, max_depth=4, learning_rate=0.1, colsample_bylevel=0.8, n_jobs=-1)
config = [200, 4, 0.1, 0.8, -1]
# Timesteps = 14 for each model (14 out of 32 timesteps)
timesteps = 14

# grid search
scores = repeat_evaluate(data_train, config, timesteps)
#print("socres after repeat evaluate: %s" % scores)
# summarize scores
summarize_scores('xgbregressor', scores)


