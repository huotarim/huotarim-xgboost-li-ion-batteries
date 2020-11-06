# Scatterplot Matrix
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import to_datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'HPmth023.csv'
filename = DIR + FILE

# Use the following labels
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','t_1','SOH']
dataset = read_csv(filename, usecols=names)
data = dataset.values

# No split-out of validation dataset to test and validation sets
test_size = 0.4
train_size = None
#print(train_size, type(train_size))

# IMPORTANT: keep time series order by shuffle=False
X_train, X_test = train_test_split(data, test_size=test_size, train_size=train_size, shuffle=False)
#print(X_train)

# convert to dataframe
dfX = DataFrame(X_train)

scatter_matrix(dfX)
pyplot.show()
