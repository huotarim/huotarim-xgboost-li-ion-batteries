# Implementation of walk-forward validation for XGBoost model development
#
# Makridakis (1998) shampoo sales data set, walk-forward validation
from pandas import to_datetime
from pandas import read_csv

# Load data
DIR = './'
FILE = 'household_power_consumption.csv'
filename = DIR + FILE

# Below if swapping to file from one month interval to X interval data.
data = read_csv(filename, parse_dates=[['Date', 'Time']], na_values = ['?'])
data.dropna()
#Reorder so that Global_active_power is the last and predicted variable col 1-> last
order = [0,2,3,4,5,6,7,1]
data = data[[data.columns[i] for i in order]]
print("data.head()\n %s" % data.head())
data = data.set_index(data.Date_Time)
data = data.resample('M').mean()
data.to_csv('household_power_monthly.csv')

