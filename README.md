# IMT_ML_project
This is a project from Machine Learning Course in IMT Atlantique. It fits several regresion models based on any given dataset.


## 1. Execution
To train the models run the following command : '>python BostonHousing.py --root=data --file_name=ProstateCancer.csv --target=lpsa --categorical_names train --sep=; --test_size=0.2 --corr_ratio=0.5 ' <br />

--root : Directory path to the root folder containing the script and the datasets. Default value equal to 'data'. <br />
--file_name : Name of the dataset folder. It must be a csv file. <br />
--target : Name of the variable to predict <br />
--categorical_names : Specify all the categorical feature names that need to be encoded. <br />
--sep : Separator between values in the csv file. <br />
--test_size : Ratio of test from the whole dataframe. <br />
--corr_ratio : Ratio of Correlation between the features and the target. Features with lower correlation value will be discarded.<br />

## 2. Datasets Used in this repository
 1. Boston housing dataset: https://www.kaggle.com/altavish/boston-housing-dataset
 2. I Prostate cancer: https://web.stanford.edu/~hastie/ElemStatLearn/data.html