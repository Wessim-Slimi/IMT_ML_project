"""
           /*  This File Contains all
             *  the needed functions to run the models
             */
"""
# doing basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import os
import argparse
import json

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
from sklearn import tree





# Input1: data_path : string value     : path to the csv file
# Input2: test_size : float         : ratio of test and train
# Output1: df_train : DataFrame     : training dataframe
# Output2: df_test  : DataFrame     : testing dataframe
# @Author : Mohamed Ouassim Slimi
def read_data(data_path, sep):
    """
    Reads a dataset from a given path to a csv file and returns a dataframe
    """
    # reading dataset
    df = pd.read_csv(data_path, sep=sep)
    # no need to drop na values here, they will be imputed
    return df

# @Author : Louis Sant'Anna
def impute_missing_data(df):
    """
    Imputation for numerical and categorical value
    """
    df.replace([np.inf, -np.inf], np.nan, inplace= True)
    df.fillna(df.median(), inplace = True)
    return df


def serie_points_cloud(data_path, sep, df_target):
    df = read_data(data_path, sep)
    df = impute_missing_data(df)

    file = os.path.basename(data_path)
    file_name = os.path.splitext(file)[0]
    target = df_target[file_name]

    for column in df:
        X = df[column]
        y = df[target]

        plt.scatter(X,y)
        plt.xlabel(column)
        plt.ylabel(target)
        plt.show()

def data_description(df):
    return df.head(), df.describe()


# Input 1 : dataframe
# Input 2 : float : ratio of the testing dataset
# Output1 : Normalized, splitted and shuffled training dataset
# Output2 : Normalized, splitted and shuffled testing dataset
# @Author : Mohamed Ouassim Slimi

def normalize_shuffle_split(df, test_size):
    """
    Normalize and shuffle the data in the given dataset
    Splits the data into training and testing dataframe
    """
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

# Input 1 : dataframe
# Input 2 : Column name of the categorical data present in the dataframe
# Output  : dataframe with encoded data
# @Author : Louis Sant'Anna
def one_hot_encode(df, categorical_names):
    """
    Transform categorical features into numerical ones
    """
    # Categorical encoding
    if categorical_names == None:
        return df

    for name in categorical_names:
        domain = list(set(df[name]))
        dico = {}
        L = list(df[name])
        for i in range(len(domain)):
            dico[domain[i]]= i
        for i in range(len(L)):
            L[i] = dico[L[i]]
        df[name] = L
    return df

# Input 1 : Data Array
# @Author : Mohamed Ouassim Slimi
def log_transform_arr(X,value=0.3):
    """
    Performs a Skewness removal on a array of data
    """
    return np.log1p(X)
# # Input 1 : Dataframe
# @Author : Louis Sant'Anna
def log_transform_df(df, value=0.3):
    """
    Performs a Skewness removal on a dataframe of data
    """
    for col in df.columns:
        if np.abs(df[col].skew()) > value:
            df[col] = np.log1p(df[col])
            df[col] = log_transform_arr(df[col], value)
    return df



# Input 1 : dataframe
# Input 2 : target column name
# Input 3 : Correlation Threshold
# Output  : return column names of the highly correlated features within the target
# @Author : Mohamed Ouassim Slimi
def get_correlated_features_name(df, y_name, threshold=0.5):
    """
    returns highly correlated features
    """
    correlation_matrix = np.abs(df.corr().round(2))
    y = correlation_matrix[y_name]
    column_names = []
    d = correlation_matrix[y_name]
    for i in range(len(d)):
        if d[i]>0.5 and list(d.index)[i]!=y_name:
            column_names.append(list(d.index)[i])
    #column_names = column_names.remove(y_name)
    return column_names


# @Author : Louis Sant'Anna
def Linear_Model():
    """
    Defining Linear Model
    """
    model = linear_model.LinearRegression()
    return model

# @Author : Mohamed Ouassim Slimi
def Linear_Regression_Ridge_Model():
    """
    Defining Regression Ridge Model
    """
    model = linear_model.Ridge()
    return model

# @Author : Louis Sant'Anna
def Polynomial_Model():
    """
    Defining Polynomial Model
    """
    model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
    return model
# @Author : Mohamed Ouassim Slimi
def SVR_Model(n_splits=10,kernel='rbf',C=1e3, gamma=0.1 ):
    """
    Defining SVR Model
    """
    # Shuffeling the data
    kf = KFold(n_splits=n_splits)
    svr_regressor= SVR(kernel=kernel, C=C, gamma=gamma)
    #proceeding to GridSearch to tune hyperparameters
    model = GridSearchCV(svr_regressor, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
    return model
# @Author : Louis Sant'Anna
def Decision_Tree_Model(n_splits=10, max_depth=5):
    """
    Defining Decision Tree Model
    """
    # Shuffeling the data
    kf = KFold(n_splits=n_splits)
    desc_tr = DecisionTreeRegressor(max_depth=max_depth)
    # GridSearch to tune Hyperparameters
    model = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
    return model
# @Author : Mohamed Ouassim Slimi
def Knn_Model(n_splits=10,n_neighbors=7 ):
    """
    Defining Knn Model
    """
    # Shuffeling the data
    kf = KFold(n_splits=n_splits)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    model = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
    return model
# @Author : Louis Sant'Anna
def Gradient_Boosting_Model(n_splits=10,alpha=0.9, learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=30):
    """
    Defining GradientBoostingRegressor
    """
    # Shuffeling the data
    kf = KFold(n_splits=n_splits)
    gbr = GradientBoostingRegressor(alpha=alpha,learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=random_state)
    param_grid={'n_estimators':[100, 200], 'learning_rate': [0.1,0.05,0.02], 'max_depth':[2, 4,6], 'min_samples_leaf':[3,5,9]}
    model = GridSearchCV(gbr, cv=kf, param_grid=param_grid, scoring='neg_mean_squared_error')
    return model

# Input 1 : model
# Input 2 : training features
# Input 3 : training labels
# Input 4 : testing features
# Output : Training model
# @Author : Mohamed Ouassim Slimi
def model_fit_and_predict(model,x_train, y_train, x_test):
    """
    trains the model
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred

# @Author : Louis Sant'Anna
def RMSE(y, y_pred):
    """
    Calculates Root Mean Squared Error
    """
    return np.sqrt(np.mean((y-y_pred)**2))

# Input 1 : Dictionary of model
# Input 2 : training features
# Input 3 : testing features
# Input 4 : training labels
# Input 5 : testing labels
# Input 6 : list of strings containing name of the features
# Input 7 : String containing name of the label to predict
# @Author : Mohamed Ouassim Slimi
def fit_models(models, x_train, x_test, y_train, y_test, feature_names, class_names):
    """
    Trains multiple models on the same data
    """
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
        if model_name == 'DecisionTreeRegressor':
            depth = models[model_name].best_params_
            print(depth)
            m = DecisionTreeRegressor(max_depth=depth['max_depth'])
            m.fit(x_train, y_train)
            fig, ax = plt.subplots()
            tree.plot_tree(m, feature_names=feature_names, class_names=class_names, filled=True, fontsize=8, ax=ax)
            plt.show()
    return res, errors
# plotting results
# @Author : Louis Sant'Anna
def plot_errors(res, title):
    res.plot(alpha=0.8, title=title).get_figure().savefig(title.split('.')[0])
    plt.show()
