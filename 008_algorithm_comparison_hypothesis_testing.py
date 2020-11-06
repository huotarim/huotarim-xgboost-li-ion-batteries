## DANGER: paired_ttest_5x2cv NO DOCUMENTATION ON CV SPLITS. 
# SO PROBABLY SHUFFELS TIME SERIES DATA TO NON-TIME SERIES DATA
# spot check machine learning algorithms on a battery
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import HuberRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neural_network import MLPRegressor
#from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from mlxtend.evaluate import paired_ttest_5x2cv
# filter warnings
from warnings import catch_warnings
from warnings import filterwarnings

# file
DIR = '../Logsn/ind_and_selBcol/v140/'
BASEFILE = 'HPmth023.csv'
FILE = DIR + BASEFILE

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	# Replace existing sting column labels with numbers as otherwise stings are passed to models
	dataframe = read_csv(full_path, header=0, names=[0,1,2,3,4,6,7,8,9,10,11,12,13,14])
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	# drop serial number and timedate from the inputs
	sn = 0
	datetime = 1
	X, y = dataframe.drop([sn, datetime, last_ix], axis='columns'), dataframe[last_ix]
	#split-out to intended test and validation sets to inihibt data leak at this stage
	test_size = 0.4
	train_size = None
	# IMPORTANT: keep time series order by shuffle=False
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=False)
	return X_train.values, y_train
 
# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    tscv = TimeSeriesSplit(n_splits=5)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='explained_variance', cv=tscv, n_jobs=-1)
    return scores
 
# compare models with statistical hypothesis test
def hypothesis_testing_between_two_models(estimator1, estimator2, X, y, 
    scoring='explained_variance', random_seed=1):
    t, p = paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, 
        scoring=scoring, random_seed=random_seed)
    return t, p

# define models to test
def get_models():
	models, names = list(), list()
	# CART -> away; compared in paper I
	models.append(DecisionTreeRegressor())
	names.append('CART')
	# ADA
	models.append(AdaBoostRegressor())
	names.append('ADA')
	# Bagging
	#models.append(BaggingRegressor(n_estimators=100))
	#names.append('BAG')
	# RF -> away?; compared in paper I
	#models.append(RandomForestRegressor(n_estimators=100))
	#names.append('RF')
	# GBM -> away; looks like XGBM
	#models.append(GradientBoostingRegressor(n_estimators=100))
	#names.append('GBM')
	# XGB
	models.append(XGBRegressor(n_estimators=100))
	names.append('XGB')
	# LGBM (light gradinet boosting)  -> away; scoring values really off
	#models.append(LGBMRegressor(n_estimators=100))
	#names.append('LGBM')
	# LinR
	models.append(LinearRegression())
	names.append('LinR')
	# SGD -> away; scoring values by far off compared to others 
	#models.append(SGDRegressor())
	#names.append('SGD')
	# Huber -> away, but looks like LinR
	#models.append(HuberRegressor())
	#names.append('Huber')
	# KNN (Nearest neighbors)
	#models.append(KNeighborsRegressor())
	#names.append('KNN')
	# MLP (Perceptron) -> way off wiht this numbe of samples
	#models.append(MLPRegressor())
	#names.append('MLP')
	return models, names

# load the dataset
X, y = load_dataset(FILE)
# define models
models, names = get_models()
results = list()

# evaluate each model
for i in range(len(models)):
	# wrap the model i a pipeline
	pipeline = Pipeline(steps=[('m',models[i])])
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
	results.append(scores)

# test hypothesis on two models
# GOOD FOR CLASSIFCATION OR NON-TIMSERIES REGRESSION; 
# NO DOCUMENTATION ON CV SPLITS. SO PROBABLY SHUFFELS TIME SERIES DATA TO NON-TIME SERIES DATA
debug = False
for i in range(len(models)):
    for j in range(len(models)):
        if j <= i:
            continue
        else:
            # wrap models i, j in pipelines
            pipeline_i = Pipeline(steps=[('m',models[i])])
            pipeline_j = Pipeline(steps=[('m',models[j])])
            # show all warnings on fail on exception if debugging
            if debug:
                # Check hypothesis between two models
                t, p = paired_ttest_5x2cv(estimator1=pipeline_i, estimator2=pipeline_j, X=X, y=y, 
                scoring='explained_variance', random_seed=1)
            else:
                try:
                    with catch_warnings():
                        filterwarnings("ignore")
                        # Check hypothesis between two models
                        t, p = paired_ttest_5x2cv(estimator1=pipeline_i, estimator2=pipeline_j, X=X, y=y, 
                            scoring='explained_variance', random_seed=1)
                except:
                    error = None
            if p is not None:
                # summarize
                print('> %s explained variance %.3f (%.3f)' % (names[i], mean(results[i]), std(results[i])))
                print('> %s explained variance %.3f (%.3f)' % (names[j], mean(results[j]), std(results[j])))
                print('p-value: %.3f, t-statistic: %.3f' % (p, t))
                # interpret the result
                if p <= 0.05:
                    print('Difference between mean performance is probably real')
                else:
                    print('Algorithms probably have the same performance')




