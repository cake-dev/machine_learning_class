import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

train = pd.read_csv('data/kaggle_train.csv')
test = pd.read_csv('data/kaggle_test.csv')

# keep only our top 4 columns f4, f5, f1, f3
train = train[['id', 'target', 'f4', 'f5', 'f1', 'f3']]
test = test[['id', 'f4', 'f5', 'f1', 'f3']]

# split data into train and test sets
X = train.drop(['target', 'id'], axis=1)
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(random_state=42)

# grid search on xgb
params = {
    'n_estimators': [200, 350, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [1, 2, 3],
    'min_child_weight': [1, 2],
    'gamma': [0, 0.1],
    'subsample': [0.75, 1, 1.25],
    'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}
grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='neg_mean_squared_error', cv=5, verbose=1)
# grid = RandomizedSearchCV(estimator=xgb, param_distributions=params, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('RMSE:', np.sqrt(mse))