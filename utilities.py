# doing basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tqdm import tqdm
from sklearn import preprocessing, datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor



# This File Contains all the needed functions to run the models

def read_data(data_path, test_size):
	# reading dataset

	df = pd.read_csv(data_path)
	# removing nan values from dataset
	df = df.dropna()
	# removing outliers
	df = df[~(df['MEDV'] >= 50.0)]
	# normalizing data
	normalized_df=(df-df.min())/(df.max()-df.min())
	# Shuffeling the data
	normalized_df = shuffle(normalized_df)
	# splitting the data into training and testing
	df_train, df_test = train_test_split(normalized_df, test_size=test_size)
	# reindexing the dataframe
	df_train.index = list(range(len(df_train)))
	df_test.index = list(range(len(df_test)))
	return df_train, df_test

# Defining and tuning the models
def Linear_Model():
	model = linear_model.LinearRegression()
	return model

def Linear_Regression_Ridge_Model():
	model = linear_model.Ridge()
	return model

def Polynomial_Model():
	model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
	return model

def SVR_Model():
	# Shuffeling the data
	kf = KFold(n_splits=10)
	svr_regressor= SVR(kernel='rbf', C=1e3, gamma=0.1)
	#proceeding to GridSearch to tune hyperparameters
	model = GridSearchCV(svr_regressor, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
	return model

def Decision_Tree_Model():
	# Shuffeling the data
	kf = KFold(n_splits=10)
	desc_tr = DecisionTreeRegressor(max_depth=5)
	# GridSearch to tune Hyperparameters
	model = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
	return model

def Knn_Model():
	# Shuffeling the data
	kf = KFold(n_splits=10)
	knn = KNeighborsRegressor(n_neighbors=7)
	model = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
	return model

def Gradient_Boosting_Model():
	# Shuffeling the data
	kf = KFold(n_splits=10)
	gbr = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=30)
	param_grid={'n_estimators':[100, 200], 'learning_rate': [0.1,0.05,0.02], 'max_depth':[2, 4,6], 'min_samples_leaf':[3,5,9]}
	model = GridSearchCV(gbr, cv=kf, param_grid=param_grid, scoring='neg_mean_squared_error')
	return model

# fit a model
def model_fit_and_predict(model,x_train, y_train, x_test):
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	return y_pred

def RMSE(y, y_pred):
	return np.sqrt(np.mean((y-y_pred)**2))

def fit_models(models, x_train, x_test, y_train, y_test):
	# dataframe that will contain predictions for each model
	res = pd.DataFrame()
	res['GroundTruth'] = y_test
	# Dictionary that will contain value errors for each model
	errors = {} 
	# fitting the models and making predictions
	for model_name in models.keys():
		print('fitting model : {}'.format(model_name))
		res[model_name] = model_fit_and_predict(models[model_name], x_train, y_train, x_test)
		# calculating RMSE of each model
		errors[model_name] =  RMSE(y_test, res[model_name].values)
	# plotting results
	plt.figure(figsize=(40,10))
	res.plot(alpha=0.5)
	plt.show()
	return res, errors
