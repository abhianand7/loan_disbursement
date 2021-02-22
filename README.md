Approach
1. Model Selection: xgboost
    *  Ensemble technique to handle multiple features and have least amount of correlation between the weak learners
2. Data Pre-processing Steps:
    *  Remove features with missing values larger than the threshold value(0.7) for the entire dataset
    *  Clean dataset
    *  Handle the missing values for numerical features
    *  Fill numerical missing values with median of the features.
    *  Not using mean, in order to reduce the effect of outliers.
    *  Handle the missing values for Categorical features
    *  Used the most frequent value(mode) of each categorical feature to fill the missing values.
    *  Encoding categorical features using Target Encoding(Mean Encoding) for encoding the categories
    *  useful when the cardinality of categorical variable is very high.
    *  Though susceptible to over-fitting.
3.  Using SMOTE to remove any imbalance in the dataset
4.  Plotting and observing correlation matrix between input features

##### Xgboost Model
parameters used
{'max_depth': 8, 'n_estimators'=800, 'eta': 0.001, 'objective': 'binary:logistic', 'gamma': 1, 'subsample': 0.8, 'colsample_bytree': 0.8}
train and test split used during training: 75:25

##### Result Obtained:
F1 socre obtained: 0.96 (Evaluation AUC score on test set)
Matthews Coefficient Score: 0.9284

#### Next steps
Further Improvements:
1. Prepare a proper data pipeline so that the data pre-processings can be applied without any hassle.
2. Improvements on categorical features:
    *  create hypothesis by doing anomaly detection and multivariate gaussian distribution analysis of the data
    *  handling less frequent categorical variables properly
    *  apply regularization when using target encoding method for categorical features, which will result in reduced overfitting
    *  gridsearch for finding more optimal parameters for xgboost model
    *  overall feature engineering requires more attention
3. One of the biggest drawback of handling the missing values by replacing it with mean, median, mode is that they become sensitive to outliers and introduces more noise in the data.
    * Best way is to predict the missing values rather than filling it using statistical methods
    * Use DataWig to impute missing values using deep learning


### Github Project Links
1. https://github.com/abhianand7/ReduceNPA
2. https://github.com/abhianand7/TextClassification/blob/master/main.py
3. https://github.com/abhianand7/nli-classifier
4. https://github.com/abhianand7/sales-forecast
