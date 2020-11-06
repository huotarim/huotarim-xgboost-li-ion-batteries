# k-fold cross validation evaluation of xgboost model
import  pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from xgboost import plot_tree
from matplotlib import pyplot
#from sklearn.metrics import explained_variance_score

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
FILE = 'JPmth023.csv'
filename = DIR + FILE

# Attach labels
#names= ['Wh_sum', 'fD_sel', 'SOH']
names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
    'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']
# Load dataset; expecting 0:2 = 2+1 columns, where the last is the explained SOH.
dataset = pd.read_csv(filename, usecols=names) 
print("dataset", dataset)

# split data into X and y
# Split-out validation dataset to test and validation sets; last is SOH
array = dataset.values
X = array[:,0:11]
Y = array[:,11]

# CV model
model = XGBRegressor()
model.fit(X,Y)

# Explanation: 
# The performance the ensemble is defined by the contribution of all trees in the ensemble. 
# The performance of one tree on the problem does not make sense on the problem 
# Some developers are very interested in getting a feeling 
# what the individual trees are doing to help better understand the whole.

# 5th and first decision trees
plot_tree(model, num_trees=4)
plot_tree(model, num_trees=0, rankdir = 'LR')
pyplot.show()