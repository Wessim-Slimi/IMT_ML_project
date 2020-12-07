# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:28:39 2020

@author: santa
"""

from utilities import *

df_target = {
    "HousingData" : "MEDV",
    "ProstateCancer" : "lpsa"
    }

root = "C:/Users/santa/Documents/MCE Courses/Machine Learning/project/IMT_ML_project/data/"
file_name = "ProstateCancer.csv"
data_path  = os.path.join(root,file_name)
sep = ";"
df_target = {
    "HousingData" : "MEDV",
    "ProstateCancer" : "lpsa"
    }
df = read_data(data_path, sep)
#serie_points_cloud(data_path, sep, df_target)
head, desc = data_description(df)
#print(head, desc, df)

new_df = df.dropna()
#print(new_df)
print(new_df.min())

"""
df = read_data(data_path, sep)
#print(df.iloc[:,len(df.columns)-1])
#print(df.loc[:,'lpsa'])
print(df[:,'lpsa'])
"""
get_correlated_features_name(df, y_name, threshold=0.5)