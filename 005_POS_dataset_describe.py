import pandas as pd
from pandas import read_csv

# Load data
INPUT_DIR = "../Logsn/ind_and_selBcol/v140/"
NAME = 'HPmth023.csv'
_filename = INPUT_DIR + NAME
# load the csv file as a data frame
pset = pd.read_csv(_filename, usecols=['TimeDate', 'BatteryStateOfCharge_Percent', 
'BatteryVoltage_V','A_mean', 'min', 'Wh_sum', 'DSOC','DV', 'fD_all', 'fD_sel', 
'cyc', 'TemperatureEnvironment_C','t_1', 'SOH'])

# descriptions
pdesc = pset.describe
print(pset.describe())
print(pset.head())

