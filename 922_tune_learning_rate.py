# battery 023, parallel cross validation
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
model = XGBRegressor(nthread=-1)
n_estimators = [50,100,200,300,400,500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
tscv = TimeSeriesSplit(n_splits=5) # 5 or 9 yields the same result (tscv splits look different, though)
#grid_search = GridSearchCV(model, param_grid, scoring="explained_variance", cv=tscv, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, scoring="explained_variance", cv=tscv, n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best evaluated variance score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# plot results
scores = numpy.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Explained variance')
pyplot.savefig('n_estimators_vs_learning_rate.png')
# pyplot.show() #  Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.