# XGBoost baseline prediction
#
# battery 023, parallel cross validation
from pandas import read_csv
from pandas import to_datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from matplotlib import pyplot

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
y = array[:,11]
# Split-out to test and validation sets
test_size = 0.4
# Keep time series order by shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=test_size, shuffle=False)

# Fit model
model = XGBRegressor()
model.fit(X_train, y_train)
print(model)

predictions = model.predict(X_test)

# Print a summary using library function for one MSE
RMSE = sqrt(mean_squared_error(y_test, predictions))
print("XGBOOST for %s RMSE: %.3f " % (FILE, RMSE))

# Explained variance for the predictions against the test1 set
r2 = r2_score(y_test, predictions)
print('Explained variance R^2 %s' % r2)

# plot predictions and expected results
pyplot.plot(y_train)
pyplot.plot([None for i in y_train] + [x for x in y_test])
pyplot.plot([None for i in y_train] + [x for x in predictions])
pyplot.show()

'''''
validation_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=validation_size, random_state=seed, shuffle=False)
# Fit regression model
params = {'n_estimators': 100, 'max_samples': 0.6, 'max_features': 1.0}
clf = ensemble.BaggingRegressor(**params)
clf.fit(X_train, y_train)

# prepare the selected model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = BaggingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
RMSE = sqrt(mean_squared_error(y_test, predictions))
print("BAG RMSE: ", RMSE)

# Rename and catch some numpy arrays for plotting
train = y_train
val = y_test
pred = predictions

# Print predictions on screen
predictions = pd.Series(predictions)
y_test = pd.Series(y_test)
for i in range(len(predictions)):
    print('>Predicted=%.3f, Expected=%.3f' % (predictions[i], y_test[i]))

# Plot predictions
x_axis_scale = range(0, val.size +1, 1)
l1, = pyplot.plot(val, 'C0', label = "expected y")
l2, = pyplot.plot(pred, 'C1--', label = "predicted $\hat y$")
pyplot.legend(handles=[l2,l1])
pyplot.title('Battery A SoH predictions by bagging', y= 1, loc = 'right')
pyplot.xticks(x_axis_scale)
pyplot.xlabel('Time (months)')
pyplot.ylabel('SoH %')
pyplot.savefig('Fig_A_SoH_pred_BAG.pdf')
pyplot.show()

# Plot all training data and predictions
x_axis_scale = range(0, train.size + val.size +1, 5)
l1, = pyplot.plot([None for i in train] + [x for x in val], 'C0^', label = "expected y")
l2, = pyplot.plot([None for i in train] + [x for x in pred], 'C1--', label = "predicted $\hat y$")
pyplot.legend(handles=[l1,l2])
pyplot.title('Battery A SoH with bagging predictions', y= 1, loc = 'right')
pyplot.xticks(x_axis_scale)
pyplot.xlabel('Time (months)')
pyplot.ylabel('SoH (%)')
all_obs = np.concatenate((train,val), axis=None)
pyplot.plot(all_obs)
pyplot.savefig('Fig_A_SoH_all_BAG.pdf')
pyplot.show()
'''''