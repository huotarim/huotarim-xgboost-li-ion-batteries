# predict SoH of a battery wiht xgboost defined model
#
from pandas import read_csv
from pandas import to_datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from numpy import std
import matplotlib.pyplot as plt

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'JPmth023.csv'
filename = DIR + FILE

# Attach labels to columns
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']

# Below if swapping to file with one minute interval data.
raw_data = read_csv(filename, header=0)
raw_data['date']=to_datetime(raw_data['TimeDate'])
dataset = raw_data.loc[:,names]
dataset = dataset.set_index(raw_data.date)
dataset = dataset.resample('M').mean()

# Get dataset values; last is SOH
array = dataset.values
X = array[:,0:11]
#print("X", X)
y = array[:,11]
# Split-out to test and validation sets
test_size = 0.4
# Keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# Fit model
model = XGBRegressor(n_estimator=200,max_depth=4, learning_rate=0.1, n_jobs=-1)
model.fit(X_train, y_train)
print(model)

# Make predictions
predictions = model.predict(X_test)

# Print a summary using RMSE of MSEs
RMSE = sqrt(mean_squared_error(y_test, predictions))
print("XGBOOST for %s RMSE: %.3f " % (FILE, RMSE))

# Explained variance for the predictions against the test1 set
r2 = r2_score(y_test, predictions)
print('Explained variance R^2 %s' % r2)

# plot predictions and expected results
plt.plot(y_train)
plt.plot([None for i in y_train] + [x for x in y_test])
plt.plot([None for i in y_train] + [x for x in predictions])

plt.show()

