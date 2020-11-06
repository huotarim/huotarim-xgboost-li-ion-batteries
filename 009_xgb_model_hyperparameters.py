# Feature importance checking (used for various batteries and model change trials)
#
from pandas import read_csv
from pandas import to_datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from numpy import sqrt
from xgboost import XGBRegressor
from matplotlib import pyplot

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'HPmth023.csv'
filename = DIR + FILE

# Use the following labels
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','t_1','SOH']
dataset = read_csv(filename, usecols=names)
array = dataset.values
X = array[:,0:len(names)-1]
y = array[:,len(names)-1]

# Split-out validation dataset to test and validation sets
test_size = 0.4
# IMPORTANT: keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# Fit model
model = XGBRegressor()
model.fit(X_train, y_train)
print(model)

# Make predictions for test data
predictions = model.predict(X_test)

# Evaluate predictions
evc = explained_variance_score(y_test, predictions)
print("Explained variance: %.2f%%" % (evc * 100.0))
#mae = mean_absolute_error(y_test, predictions)
#print("Mean absolute error: %.2f" % (mae)) 
rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %.2f" % (rmse)) 
r2 = r2_score(y_test, predictions)
print("R2: %.2f" % (r2)) 