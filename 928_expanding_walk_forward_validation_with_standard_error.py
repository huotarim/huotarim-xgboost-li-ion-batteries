# Implementation of walk-forward validation for XGBoost model development
#
# Makridakis (1998) shampoo sales data set, walk-forward validation
from pandas import to_datetime
from xgboost import XGBRegressor
from numpy import array
from pandas import DataFrame
from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import array
from scipy.stats import sem
#from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

# 1st split: dataset into train1/ test1 sets 
def train_test_split_all(data, n_testratio):
    n_test = int(len(data) * n_testratio)
    return data[:-n_test], data[-n_test:]

# 2nd split: original train1 into train_X, train_y, test_X, test_y for walk-forward
def train_validation_split(data, timesteps, n):
    # input sequence (t-timesteps, ... t-1); forecast sequence t
    n_train_start = 0 # only line of data to cvhange ifor expanding window 
    n_train_stop = n + timesteps
    train_X = array(data[n_train_start : n_train_stop, 0 : -1])
    train_y = array(data[n_train_start : n_train_stop, -1])
    test_X = array(data[n_train_stop, 0 : -1])
    test_y = array(data[n_train_stop, -1])
    #print("train_X %s, test_X %s, train_y %s, test_y %s" % (train_X, test_X, train_y, test_y))
    return train_X, test_X, train_y, test_y

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt((actual - predicted) * (actual - predicted))

# fit a model
def model_fit(train_X, train_y, config):
    # unpack config
    n_estimator, max_depth, learning_rate, colsample_bytree, n_jobs = config
    # define model
    model = XGBRegressor(n_estimator=n_estimator, max_depth=max_depth, 
    learning_rate=learning_rate, colsample_bytree=colsample_bytree, n_jobs=n_jobs)
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
    train_X, _, train_y, test_y = train_validation_split(data, timesteps, data_head_index)
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
    #print(' > %.3f' % error)
    return error

# repeat evaluation of a config; default 5 times (as of now dataset has only few = 32 timesteps...)
def repeat_evaluate(data, config, timesteps, n_repeats=5):
    # fit and evaluate the model n times indicated by n_repeats; i is offset to drop history from the head 
    scores = [walk_forward_validation(data, timesteps, i, config) for i in range(n_repeats)]
    return scores

# summarize model performance
def summarize_scores(filename, name, scores):
    # print a summary
    scores_m, score_std = mean(scores), sem(scores)
    print('%s with %s: RMSE mean=%.3f se=%.3f' % (filename, name, scores_m, score_std))
    # box and whisker plot
    #pyplot.boxplot(scores)
    #pyplot.show()

# Load data
DIR = './'
FILE = 'household_power_monthly.csv'
filename = DIR + FILE

# read and reindex
data = read_csv(filename)
#print("head()\n", data.head() )
#data = data.set_index(data.Date_Time)
data = data.drop(columns=['Date_Time'])
#print("head()\n", data.head() )

# From dataframe to individual numpy arrays with df.values 
data = data.values
#print("data.shape 0 and 1\n", data.shape[0], data.shape[1])

# Split all data to train and test; use train only for walk-forward validation
test_ratio= 0.6
data_train, _ = train_test_split_all(data, test_ratio)
print("data_train.shape 0 and 1\n", data_train.shape[0], data_train.shape[1])

# define config 
# (n_estimator, max_depth, learning_rate, colsample_bytree, n_jobs)
config = [200, 2, 0.01, 0.4, -1]
# Timesteps = n for each model (n out of data_train timesteps)
timesteps = 15

# grid search
scores = repeat_evaluate(data_train, config, timesteps)
#print("socres after repeat evaluate: %s" % scores)
# summarize scores
summarize_scores(FILE, 'xgbregressor, sliding window', scores)

#print("data_train\n", data_train )
columns = [f'col_{num}' for num in range(data_train.shape[1])]
index = [f'{num}' for num in range(data_train.shape[0])]
df = DataFrame(data_train, columns=columns, index=index)
#print("df\n", df)

#pyplot.plot('col_6', data=df)
#pyplot.show()