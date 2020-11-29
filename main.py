"""
	/* Example of execution
	 * $python main.py 
	 * --root=data 
	 * --file_name=ProstateCancer.csv 
	 * --target=lpsa 
	 * --categorical_names=train 
	 * --sep=; 
	 * --test_size=0.2 
	 * --corr_ratio=0.5 
	 */
"""

from utilities import *


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='data',
                    help="Root of the program")
parser.add_argument("--file_name", type=str,
                    help="Name of the dataset csv file")
parser.add_argument("--target", type=str,
                    help="Name of the targeted value in the dataset")
parser.add_argument("--categorical_names", nargs='+', default=[],
                    help="Name of the features that are considerated as categorical")
parser.add_argument("--sep", type=str,
                    help="Seperator between values in the csv file")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Test ratio in from the whole dataset. Default value equal to 0.2")
parser.add_argument("--corr_ratio", type=float, default= 0.5,
                    help="Correlation threshold value. Columns with correlation value lower than corr_ratio are discarded from the training dataset. Default value equal to 0.5")

args = parser.parse_args()


# defining Constants
root 	   = args.root #'data'
file_name  = args.file_name #'ProstateCancer.csv'
target     = args.target #'lpsa'
sep 	   = args.sep #';'

test_size  = args.test_size #0.2 # 80% for trainig and 20% for testing
corr_ratio = args.corr_ratio #0.5

categorical_names = args.categorical_names #['train']

data_path  = os.path.join(root,file_name)



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
df = read_data(data_path, sep=sep)

# Encoding the categorical data
df = one_hot_encode(df, categorical_names)

# Replacing nans with median
df = impute_missing_data(df)

# Normalizing the data
df_train, df_test = normalize_shuffle_split(df, test_size)



# Selecting only highly correlated features 
column_sels =  get_correlated_features_name(df_train, target, corr_ratio)
print("Selected Columns for the Model : {}".format(column_sels))

# preparing the dataset
x_train = df_train.loc[:,column_sels]
y_train = df_train[target]
x_test  = df_test.loc[:,column_sels]
y_test 	= df_test[target]



# Let's try to remove the skewness of the data through log transformation.
"""y_train =  np.log1p(y_train)
y_test =  np.log1p(y_test)
for col in x_train.columns:
    if np.abs(x_train[col].skew()) > 0.3:
        x_train[col] = np.log1p(x_train[col])
        x_test[col] = np.log1p(x_test[col])"""

y_train = log_transform_arr(y_train)
y_test 	= log_transform_arr(y_test)

x_train = log_transform_df(x_train)
x_test 	= log_transform_df(x_test)

# this function fit each model given in models
# calculates the prediction of each model
# stores errors in errors
# and plot the predictions
res, errors = fit_models(models, x_train, x_test, y_train, y_test,column_sels, target)

# plotting results
plot_errors(res, file_name)

print('*************************************************')
print('Printing Results | "MODEL" : RMSE')
print (json.dumps(errors,sort_keys=True, indent=4))



