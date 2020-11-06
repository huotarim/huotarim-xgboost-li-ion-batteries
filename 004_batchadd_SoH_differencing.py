from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import array

# main
DIR = '../Logsn/ind_and_selBcol/v140/'
files = ['JPmth013.csv','JPmth044.csv','JPmth014.csv','JPmth025.csv','JPmth056.csv'
    ,'JPmth067.csv','JPmth071.csv']

for FILE in files:
    filename = DIR + FILE
    data = read_csv(filename, parse_dates= True,squeeze=True)

    # add lagged SOH data as 2nd but last column
    a = DataFrame(data["SOH"].values)
    b = a.shift(1)
    data = data.drop(labels="SOH", axis=1)
    data["t_1"] = b
    data["SOH"] = a
    data = data.drop(axis=0,index=0)
    print("data.head()\n", data.head()) # drop 1st nan line as the regression comparison cannot handle this except for xgboost.

    #  save to H-series
    NAME = "H" + FILE[1:]
    filename = DIR + NAME
    data.to_csv(filename)

    #free memory
    del a
    del b
    del data

