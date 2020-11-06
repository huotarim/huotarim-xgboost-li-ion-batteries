# Correlation Matrix Plot (generic)
from matplotlib import pyplot
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy

# # Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
#FILE = 'JPmth023.csv'
FILE = 'HPmth023.csv'
filename = DIR + FILE

# Use columns
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','t_1', 'SOH']
dataset = read_csv(filename, usecols=names)
data = dataset.values

# No split-out of validation dataset to test and validation sets
test_size = 0.4
train_size = None

# IMPORTANT: keep time series order by shuffle=False
X_train, X_test = train_test_split(data, test_size=test_size, train_size=train_size, shuffle=False)

# convert to dataframe for plotting
dfX = DataFrame(X_train)

# Figure labels
labels = ['SoC','V','A','min',
    'Wh','DV','fDa','fDs','cyc','C','t-1', 'SoH']
# Load dataset; the last is the explained SOH.
correlations = dfX.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,12,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
pyplot.show()