# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from src.preprocessing import build_preprocessor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def load_processed_data(path):
    return pd.read_csv(path)


def split_data(df, target_col='Time_taken(min)', test_size=0.2, random_state=42):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
    }
    
    return metrics, model 


def tune_random_forest(X_train, y_train, n_iter=20, cv=5, random_state=42):
    param = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    
    rf = RandomForestRegressor(random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    return search

