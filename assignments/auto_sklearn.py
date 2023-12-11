import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# autosklearn
import autosklearn.regression

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    train = pd.read_csv('data/kaggle_train.csv')
    test = pd.read_csv('data/kaggle_test.csv')

    # keep only our top 4 columns f4, f5, f1, f3
    train = train[['id', 'target', 'f4', 'f5', 'f1', 'f3']]
    test = test[['id', 'f4', 'f5', 'f1', 'f3']]

    X = train.drop(['target', 'id'], axis=1)
    y = train['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=7200,
        per_run_time_limit=360,
        tmp_folder='../../Data/autosklearn_tmp_2',
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        n_jobs=8,
        memory_limit=None,
    )
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('RMSE:', np.sqrt(mse))

    # show the models in the ensemble
    print(automl.show_models())

    # show results
    print(automl.sprint_statistics())

    # show leaderboard
    print(automl.leaderboard())

    # inspect feature importance
    print(automl.get_models_with_weights())

    # save predictions from best model(s)
    predictions = automl.predict(test.drop('id', axis=1))
    test['target'] = predictions
    test[['id', 'target']].to_csv('data/autosklearn_predictions.csv', index=False)