# This is copied from classification; may not make sense for regression...

# use feature importance for feature selection
from pandas import read_csv
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from matplotlib import pyplot

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

# Split-out validation dataset to test and validation sets
test_size = 0.4
# IMPORTANT: keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# fit model on all training data
model = XGBRegressor()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "rmse"], eval_set=eval_set,
verbose=True)
# make predictions for test data
predictions = model.predict(X_test)
rmse2 = sqrt(mean_squared_error(y_test, predictions))
print("RMSE of predictions: %.2f%%" % (rmse2))

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot RMSE
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('RMSE')
pyplot.title('XGBoost RMSE')
pyplot.show()
