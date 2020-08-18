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
# FILE = 'JPmin023.csv'
FILE = 'JPmth023.csv'
filename = DIR + FILE

# Attach labels to columns
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']
# Load dataset; expecting 0:11 = 11+1 columns, where the last one is the explained y = SOH.
dataset = read_csv(filename, usecols=names) 
array = dataset.values
X = array[:,0:11]
y = array[:,11]

# Split-out validation dataset to test and validation sets
test_size = 0.4
# IMPORTANT: keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=False)

# Fit model over train
model = MyXGBRegressor()
model.fit(X_train, y_train)

# make predictions for test data
predictions = model.predict(X_train)
rmse = sqrt(mean_squared_error(y_train, predictions))
print("RMSE of train predictions: %.2f%%" % (rmse * 100.0))
ev = explained_variance_score(y_train, predictions)
print("Explained variance of train predictions: %.2f" % (ev))

# Fit model over train
model2 = MyXGBRegressor()
model2.fit(X_test, y_test)

# make predictions for test data
predictions2 = model2.predict(X_test)
rmse2 = sqrt(mean_squared_error(y_test, predictions2))
print("RMSE of test predictions: %.2f%%" % (rmse2 * 100.0))
ev2 = explained_variance_score(y_test, predictions2)
print("Explained variance of test predictions: %.2f" % (ev2))

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

# Plot importances
plot_importance(model)
plot_importance(model2)
pyplot.show()

