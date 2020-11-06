# Plot 2 x 2 graph for 4 batteries' data that have passed Wilcoxon test
#
# batteries 044, 056, 067 and 071 a.k.a. b,c,d,e.
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score 
from numpy import sqrt
from numpy import std
from numpy import linspace
import matplotlib.pyplot as plt

# Load data
DIR = '../Logsn/ind_and_selBcol/v140/'
#FILEa = 'HPmth023.csv'
FILEb = 'HPmth044.csv'
FILEc = 'HPmth056.csv'
FILEd = 'HPmth067.csv'
FILEe = 'HPmth071.csv'
FILENAMES = [DIR + FILEb, DIR + FILEc, DIR + FILEd, DIR + FILEe]
GRAPHNAMES = ['b', 'c', 'd', 'e']
GRIDPOSITION = [[0,0], [0,1], [1,0], [1,1]]

# 4 subplots in grid of 2x2
fig, axs = plt.subplots(2,2)

# Set i for indexing names and positions
i = 0

for _filename in FILENAMES:

    # Use columns 
    names = ['BatteryStateOfCharge_Percent','A_mean','Wh_sum','DV','t_1','SOH']

    # Preprosessing; valid for minutely or monthly interval data.
    dataset = read_csv(_filename, usecols=names)

    # Get dataset values; last is SOH
    array = dataset.values
    X = array[:,0:len(names)-1]
    y = array[:,len(names)-1]

    # Split-out to test and validation sets
    test_size = 0.4
    # Keep time series order by shuffle=False
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=False)

    # create linear regression object 
    reg = linear_model.LinearRegression() 
    
    # train the model using the training sets 
    reg.fit(X_train, y_train) 

    # Make predictions for test data
    predictions = reg.predict(X_test)

    # Evaluate predictions
    evc = explained_variance_score(y_test, predictions)
    print("Explained variance: %.2f%%" % (evc * 100.0))
    mae = mean_absolute_error(y_test, predictions)
    print("Mean absolute error: %.2f" % (mae)) 
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print("RMSE: %.2f" % (rmse)) 
    r2 = r2_score(y_test, predictions)
    print("R2: %.2f\n" % (r2)) 

    # 2 * standard deviation as error 
    error = 2 * std(predictions)

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


