# Scrutinize feature importance (for a possible selection of subset of features)
#
from pandas import read_csv
from numpy import sqrt
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot

# define custom class to fix bug in xgboost 1.0.2
class MyXGBRegressor(XGBRegressor):
    @property
    def coef_(self):
        return None

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'HPmth023.csv'
filename = DIR + FILE

# Select used columns based on the Pearson correlation results
#names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
#    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V', 'A_mean', 
    'Wh_sum','DV','fD_sel','cyc','TemperatureEnvironment_C','t_1', 'SOH']

# Load dataset; the last one is the explained y = SOH.
dataset = read_csv(filename, usecols=names) 
array = dataset.values
X = array[:,0:len(names)-1]
y = array[:,len(names)-1]

# Split-out validation dataset to test and validation sets
test_size = 0.4
# IMPORTANT: keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=False)

# Fit model over train
model = MyXGBRegressor()
model.fit(X_train, y_train)

# Fit model over test
#model2 = MyXGBRegressor()
#model2.fit(X_test, y_test)

# Fit train model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    
    # train model
    selection_model = XGBRegressor()
    selection_model.fit(select_X_train, y_train)
    
    # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    ev = explained_variance_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, EV train: %.2f RMSE %.2f%%" % (thresh, select_X_train.shape[1],
        ev, rmse))

print("\n Model derived from (train) data:\n")
print(model)
# Plot importance based on fitted trees on two different ways:
# Print and plot importances; default is gain for model.feature_importances_
#print(model.feature_importances_)
# plot
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()

# A comment 23 June 2020: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
# gain over weight because gain reflects the feature’s power of grouping similar instances into a more homogeneous child node at the split.

# Fit model over test
model2 = MyXGBRegressor()
model2.fit(X_test, y_test)

# Print and plot importances for test; default is weight for importance_type
print("\n Test model (verification) NOT train:\n", model2)
plot_importance(model2, importance_type="gain", 
    title="Feature importance (test)", xlabel="F score (gain based)")

# Print and plot importances for TRAIN (main point here)
print("\n Model derived from train data:\n", model)
print(model)
plot_importance(model, importance_type="gain", 
    title="Feature importance (train)", xlabel=" F score (gain based)")
pyplot.show()