# Battery A/ 023 SoH prediction vs. observations with Adaboost
from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import std
from numpy import sqrt
from numpy import linspace
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
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)
print(model)

# Make predictions
predictions = model.predict(X_test)

# Evaluate predictions
evc = explained_variance_score(y_test, predictions)
print("Explained variance: %.2f%%" % (evc * 100.0))
mae = mean_absolute_error(y_test, predictions)
print("Mean absolute error: %.2f" % (mae)) 
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %.2f" % (rmse)) 
r2 = r2_score(y_test, predictions)
print("R2 coefficient of determination: %.2f" % (r2)) 

# For sampled population confidence interval = standard error * 1.96; https://en.wikipedia.org/wiki/Standard_error
# Here used 2 * std
error = std(predictions)*2

# Upper and lower bound for error
ne = predictions - error
pe = predictions + error

# Do not plot

