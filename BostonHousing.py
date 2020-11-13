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


from utilities import *



# defining Constants
root 	  = 'data'
file_name = 'HousingData.csv'
data_path = os.path.join(root,file_name)
test_size = 0.2 # 80% for trainig and 20% for testing
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE'] # see notebook to know why these changes
models = {
			'linear_regression'         : Linear_Model(),
			'linear_ridge_regression'   : Linear_Regression_Ridge_Model(),
			'Polynomial_Regression'     : Polynomial_Model(),
			'SVR'                       : SVR_Model(),
			'DecisionTreeRegressor'     : Decision_Tree_Model(),
			'knn'                       : Knn_Model(),
			#'GradientBoostingRegressor' : Gradient_Boosting_Model()
		 }

# Reading data and splitting trainig and testing
df_train, df_test = read_data(data_path, test_size)

# preparing the dataset
x_train = df_train.loc[:,column_sels]
y_train = df_train['MEDV']
x_test = df_test.loc[:,column_sels]
y_test = df_test['MEDV']

# Let's try to remove the skewness of the data through log transformation.
y_train =  np.log1p(y_train)
y_test =  np.log1p(y_test)
for col in x_train.columns:
    if np.abs(x_train[col].skew()) > 0.3:
        x_train[col] = np.log1p(x_train[col])
        x_test[col] = np.log1p(x_test[col])

# this function fit each model given in models
# calculates the prediction of each model
# stores errors in errors
# and plot the predictions
res, erros = fit_models(models, x_train, x_test, y_train, y_test)




