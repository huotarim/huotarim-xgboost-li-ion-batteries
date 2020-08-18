# Plot 2 x 2 graph for 4 batteries' data that have passed Wilcoxon test
#
# batteries 044, 056, 067 and 071 a.k.a. b,c,d,e.
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from numpy import std
from numpy import zeros
from numpy import linspace
from numpy import concatenate
from numpy import NaN
import matplotlib.pyplot as plt

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/JPmth/'
#FILEa = 'JPmth023.csv'
FILEb = 'JPmth044.csv'
FILEc = 'JPmth056.csv'
FILEd = 'JPmth067.csv'
FILEe = 'JPmth071.csv'
FILENAMES = [DIR + FILEb, DIR + FILEc, DIR + FILEd, DIR + FILEe]
GRAPHNAMES = ['b', 'c', 'd', 'e']
GRIDPOSITION = [[0,0], [0,1], [1,0], [1,1]]

# 4 subplots in grid of 2x2
fig, axs = plt.subplots(2,2)

# Set i for indexing names and positions
i = 0

for _filename in FILENAMES:

    # Attach labels to columns
    names = ['BatteryStateOfCharge_Percent','BatteryVoltage_V','A_mean','min',
        'Wh_sum','DSOC','DV','fD_all','fD_sel','cyc','TemperatureEnvironment_C','SOH']

    # Preprosessing; valid for minutely or monthly interval data.
    raw_data = read_csv(_filename, header=0)
    raw_data['date']=to_datetime(raw_data['TimeDate'])
    dataset = raw_data.loc[:,names]
    dataset = dataset.set_index(raw_data.date)
    dataset = dataset.resample('M').mean()

    # Get dataset values; last is SOH
    array = dataset.values
    X = array[:,0:11]
    #print("X", X)
    y = array[:,11]
    # Split-out to test and validation sets
    test_size = 0.4
    # Keep time series order by shuffle=False
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=False)

    # Fit model
    model = XGBRegressor(n_estimator=200,max_depth=4, learning_rate=0.1, 
        colsample_bylevel=0.8, n_jobs=-1)
    model.fit(X_train, y_train)
    #print(model)

    # Make predictions
    predictions = model.predict(X_test)

    # Print a summary using RMSE of MSEs
    RMSE = sqrt(mean_squared_error(y_test, predictions))
    print("XGBOOST for %s RMSE: %.3f " % (_filename, RMSE))

    # Explained variance for the predictions against the test1 set
    r2 = r2_score(y_test, predictions)
    print('Explained variance R^2 %s' % r2)

    # 1 * standard deviation as error 
    error = std(predictions)

    # Upper and lower bound for error
    ne = predictions - error
    pe = predictions + error

    # define x-axis xe for error plotting
    plen = len(predictions)
    xe = linspace(0, plen-1, plen)
    
    xi = GRIDPOSITION[i][0]
    yi = GRIDPOSITION[i][1]
    name_i = GRAPHNAMES[i]

    # plot a subplot
    axs[xi, yi].fill_between(xe, ne, pe, alpha=0.5, color='royalblue')
    axs[xi, yi].plot(y_test, color='chocolate')
    axs[xi, yi].plot(predictions, color = 'royalblue')
    axs[xi, yi].set_title('battery (%s)' % name_i, loc ='left')
    if xi == 0: 
        axs[xi, yi].text(0, 92.5, '$R^2=%.2f$'% round(r2,2))
    elif yi == 0:
        axs[xi, yi].text(0, 94.5, '$R^2=%.2f$'% round(r2,2))
    else: 
        axs[xi, yi].text(0, 93, '$R^2=%.2f$'% round(r2,2))

    # increment i for next step
    i = i+1

# pad spacing
fig.tight_layout(pad = 3)    

# Make overall labels and titles
fig.suptitle('Predicted vs. actual SoH ')

# Show labels and scale for all; hide obscures scale interpretation
for ax in axs.flat:
    ax.set(xlabel='Time (months)', ylabel='SoH (%)')
#for ax in axs.flat:
#    ax.label_outer()

# Save the result and show it
name = 'Fig_bcde_SoH_pred_stddev_XGB.pdf'
plt.savefig(name)
plt.show()


