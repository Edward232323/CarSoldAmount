import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def compute_r_squared(data: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    """
    Compute the R-squared coefficient between all variables and the target variable.

    Parameters:
    data (pandas.DataFrame): The input DataFrame
    target_variable (str): The target variable for R-squared computation

    Returns:
    pandas.DataFrame: A DataFrame containing the R-squared coefficients for each variable
    """
    r_squared_results = {}

    # Convert non-numeric columns to numeric using one-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    for column in data_encoded.columns:
        if column != target_variable:
            X = data_encoded[[column]].dropna()
            y = data_encoded[target_variable].dropna()
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if X.shape[0] > 1 and y.shape[0] > 1:  # Ensure there are enough data points
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r_squared = r2_score(y, y_pred)
                r_squared_results[column] = r_squared

    r_squared_df = pd.DataFrame(list(r_squared_results.items()), columns=['Variable', 'R-squared'])
    return r_squared_df


def get_missing_percentage(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the count and missing data percentage for each column in a DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame

    Returns:
    pandas.DataFrame: A DataFrame containing the count and missing data percentage for each column
    """
    total_rows = data.shape[0]
    missing_count = data.isnull().sum()
    missing_percentage = (missing_count / total_rows) * 100
    missing_data_df = pd.DataFrame({'Count': missing_count, 'Missing Data Percentage': missing_percentage})

    return missing_data_df
