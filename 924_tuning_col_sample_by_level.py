# battery 023, tune column subsampling by tree
from pandas import read_csv
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy

# Load data
DIR = './'
FILE = 'household_power_monthly.csv'
filename = DIR + FILE

# Use column labels 
names = ['Global_reactive_power','Voltage','Global_intensity',
    'Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_active_power']

dataset = read_csv(filename, usecols=names) 
array = dataset.values
X = array[:,0:len(names)-1]
y = array[:,len(names)-1]

# Split-out validation dataset to test and validation sets
test_size = 0.4
# IMPORTANT: keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# Grid search
model = XGBRegressor()
colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(colsample_bylevel=colsample_bylevel)
tscv = TimeSeriesSplit(n_splits=5) # 5 or 9 yields the same result (tscv splits look different, though)
grid_search = GridSearchCV(model, param_grid, scoring="explained_variance", cv=tscv, n_jobs=-1)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# plot results
pyplot.errorbar(colsample_bylevel, means, yerr=stds)
pyplot.title("XGBoost column_sample_bylevel vs Explained variance")
pyplot.xlabel('column_sample_bylevel')
pyplot.ylabel('Explained variance')
pyplot.savefig('column_sample_bylevel.png')