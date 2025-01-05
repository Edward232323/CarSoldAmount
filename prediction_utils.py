import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List
from dataclasses import dataclass
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float


def load_df_into_dmatrix(df: pd.DataFrame) -> pd.DataFrame:
    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    To encode the categorial columns, so that it can be used in regression model.
    Args:
        df: data

    Returns: processed data with encoded categorical columns

    """

    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df

def fit_model(X_train: List[str], y_train: str, **params):
    model = xgb.XGBRegressor(**params)
    model.fit(X=X_train, y=y_train)
    return model


def predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.Series, **params) -> list:
    """
    Predict the default status
    :param X: X variables
    :param y: responding variable to be predicted
    :return: the predictions of default or not
    """
    model = fit_model(X_train, y_train, **params)
    y_predict = model.predict(X_test)
    return y_predict

def evaluate(y_predict: pd.Series, y_test: pd.Series) -> EvaluationMetrics:
    """
    Evaluate the predictions using RMSE and MAE.

    Parameters:
    y_predict (pd.Series): The predicted values.
    y_test (pd.Series): The true values.

    Returns:
    dict: A dictionary containing RMSE and MAE.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mae = mean_absolute_error(y_test, y_predict)

    return EvaluationMetrics(rmse=rmse, mae=mae)