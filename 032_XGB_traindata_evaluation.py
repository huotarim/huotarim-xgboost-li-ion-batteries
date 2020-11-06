# Battery A SoH prediction vs. observations with XGBoost

# battery 023, prediction vs. observations
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import std
from numpy import sqrt
from numpy import zeros
from numpy import linspace
from numpy import concatenate
from numpy import NaN
import matplotlib.pyplot as plt

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'HPmth023.csv'
filename = DIR + FILE

# Use column labels 
names = ['BatteryStateOfCharge_Percent','A_mean','Wh_sum','DV','t_1','SOH']
dataset = read_csv(filename, usecols=names) 
array = dataset.values
X = array[:,0:len(names)-1]
y = array[:,len(names)-1]

# Split-out to test and validation sets
test_size = 0.4
# Keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# Fit model
model = XGBRegressor(n_estimator=200, max_depth=2, learning_rate=0.1, 
    colsample_bylevel=0.8, n_jobs=-1)
model.fit(X_train, y_train)
print(model)

# Split-out TRAIN set to test and validation sets for bias-variance estimation
# HAstei 2017 page 218; Brownlee (jossakin)
test_size = 0.4
# Keep time series order by shuffle=False
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train,
test_size=test_size, shuffle=False)

# Make predictions
predictions2 = model.predict(X_test2)

# Evaluate predictions
evc = explained_variance_score(y_test2, predictions2)
print("TRAIN Explained variance: %.2f%%" % (evc * 100.0))
mae = mean_absolute_error(y_test2, predictions2)
print("TRAIN Mean absolute error: %.2f" % (mae)) 
rmse = sqrt(mean_squared_error(y_test2, predictions2))
print("TRAIN RMSE: %.2f" % (rmse)) 
r2 = r2_score(y_test2, predictions2)
print("TRAIN R2 coefficient of determination: %.2f" % (r2)) 



