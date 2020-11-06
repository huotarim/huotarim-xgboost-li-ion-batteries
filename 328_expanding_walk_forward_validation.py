# Implementation of AdaBoost model verification. Walk-forward sliding window for getting CI for RMSE 
#
# battery 023, walk-forward validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from numpy import array
from pandas import DataFrame
from pandas import read_csv
from numpy import mean
from numpy import std
from scipy.stats import sem
#from sklearn.metrics import mean_squared_error
from numpy import sqrt
from matplotlib import pyplot

# 1st split: all dataset into train1/ test1 sets X = train_test_split_all(X, test_ratio)
def train_test_split_all(data, n_testratio):
    n_test = int(len(data) * n_testratio)
    return data[:-n_test], data[-n_test:]

# 2nd split: train1 into train_X, train_y, test_X, test_y for walk-forward
def train_validation_split(data, timesteps, n):
    # input sequence (t-timesteps, ... t-1); forecast sequence t
    n_train_start = 0 # 0 = expanding; n = sliding
    n_train_stop = n + timesteps
    train_X = array(data[n_train_start : n_train_stop, 0 : -1])
    train_y = array(data[n_train_start : n_train_stop, -1])
    test_X = array(data[n_train_stop, 0 : -1])
    test_y = array(data[n_train_stop, -1])
    #print("train_X %s, test_X %s, train_y %s, test_y %s" % (train_X, test_X, train_y, test_y))
    return train_X, test_X, train_y, test_y

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean((actual - predicted) **2))

# fit a model
def model_fit(train_X, train_y, config):
    # unpack config
    n_estimator, max_depth, learning_rate = config 
    # Fit model
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimator, learning_rate=learning_rate)
    model.fit(train_X, train_y)
    return model

# forecast with a pre-fit model
def model_predict(model, history):
    yhat = model.predict(history)
    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, timesteps, data_head_index, cfg):
    # split dataset; in walk-foirward validation test is typically 1.
    train_X, _, train_y, test_y = train_validation_split(data, timesteps, data_head_index)
    # fit model
    model = model_fit(train_X, train_y, cfg)
    # seed history with training dataset 
    history = train_X
    # fit model and make forecast for history
    yhat = model_predict(model, history)
    # estimate prediction error
    error = measure_rmse(yhat, test_y)
    #print(' > %.3f, %.3f,%.3f ' % (error, yhat, test_y))
    return error

# repeat evaluation of a config; default 5 times (as of now dataset has only few = 32 timesteps...)
def repeat_evaluate(data, config, timesteps, n_repeats=5):
    # fit and evaluate the model n times indicated by n_repeats; i is offset to drop history from the head 
    scores = [walk_forward_validation(data, timesteps, i, config) for i in range(n_repeats)]
    return scores

# summarize model performance
def summarize_scores(filename, name, scores):
    # print a summary
    scores_m, score_sem, score_std = mean(scores), sem(scores), std(scores)
    #print('%s with %s: RMSE mean=%.3f se=%.3f (std=%.3f)' % (filename, name, scores_m, score_sem, score_std))
    print('%s: The error of the model was %.3f +/- %.3f at the 95%% confidence level.' % (filename, scores_m, 1.96*score_sem))
    # box and whisker plot
    #pyplot.boxplot(scores)
    #pyplot.show()

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
files = ['HPmth023.csv']
for FILE in files:
    filename = DIR + FILE
    
    # Use columns and read data
    names = ['BatteryStateOfCharge_Percent','A_mean','Wh_sum','DV','t_1','SOH']
    dataset = read_csv(filename, usecols=names)
    data = dataset.values

    # Split all data to train and test for calculating walk-forward steps
    test_ratio= 0.4
    data_train, data_test = train_test_split_all(data, test_ratio)

    # timesteps = use number of timesteps equalling and starting from original train set for final verification
    timesteps = 14

    # define config 
    config = [200, 2, 0.1]

    # walk forward to get conficende intervals
    scores = repeat_evaluate(data, config, timesteps)

    # verify and get confidence intervals
    scores = repeat_evaluate(data, config, timesteps)

    # summarize scores
    summarize_scores(FILE, 'adaboost regressor validation, expanding window', scores)
