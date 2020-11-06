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

# define x-axis for plotting
plen = len(predictions)
x = linspace(0, plen-1, plen)
plt.fill_between(x, ne, pe, alpha=0.5, color='royalblue', label='+/- 2 x std')
plt.plot(y_test, color='chocolate', label='Battery (a) SoH observations')
plt.plot(predictions, color = 'royalblue', label='Battery (a) SoH predictions')
plt.title('Predicted vs. actual SoH ')
plt.legend(loc='upper center')
plt.xlabel('Time (months)')
plt.ylabel('SoH (%)')
plt.text(10.5, 95, '$R^2=%.2f$'% round(r2,2))
plt.savefig('Fig_a_SoH_pred_stddev_XGB.pdf')
plt.show()

# Rename and catch some numpy arrays for plotting
train = y_train
val = y_test
pred = predictions

# Plot predictions
x_axis_scale = range(0, val.size +1, 1)
l1, = plt.plot(val, 'C0', label = "expected y")
l2, = plt.plot(pred, 'C1--', label = "predicted $\hat y$")
plt.legend(handles=[l2,l1])
plt.title('Battery (a) SoH predictions by XGBoost', y= 1, loc = 'right')
plt.xticks(x_axis_scale)
plt.xlabel('Time (months)')
plt.ylabel('SoH %')
plt.text(10.5, 95, '$R^2=%.2f$'% round(r2,2))
plt.savefig('Fig_a_SoH_pred_XGB.pdf')
plt.show()

# Plot all training data and predictions
x_axis_scale = range(0, train.size + val.size +1, 5)
l1, = plt.plot([None for i in train] + [x for x in val], 'C0^', label = "expected y")
l2, = plt.plot([None for i in train] + [x for x in pred], 'C1--', label = "predicted $\hat y$")
plt.legend(handles=[l1,l2])
plt.title('Battery (a) SoH XGBoost predictions', y= 1, loc = 'right')
plt.xticks(x_axis_scale)
plt.xlabel('Time (months)')
plt.ylabel('SoH (%)')
all_obs = concatenate((train,val), axis=None)
plt.plot(all_obs)
plt.savefig('Fig_A_SoH_all_XGB.pdf')
plt.show()
