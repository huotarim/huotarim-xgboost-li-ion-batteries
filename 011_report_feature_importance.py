# Feature importance checking (used for various batteries and model change trials)
#
from pandas import read_csv
from pandas import to_datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from xgboost import XGBRegressor
from matplotlib import pyplot

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'JPmin023.csv'
filename = DIR + FILE

# Attach labels to columns
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']

raw_data = read_csv(filename, header=0)
raw_data['date']=to_datetime(raw_data['TimeDate'])
dataset = raw_data.loc[:,names]
dataset = dataset.set_index(raw_data.date)
#dataset = dataset.resample('2W').mean()

# Load dataset; expecting 0:11 = 11+1 columns, where the last one is the explained y = SOH.
#dataset = read_csv(filename, usecols=names) 
array = dataset.values
X = array[:,0:11]
y = array[:,11]

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