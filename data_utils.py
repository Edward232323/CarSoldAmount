import pandas as pd
from dataclasses import dataclass

@dataclass
class Data:
    X: pd.DataFrame
    y: pd.DataFrame

def read_rpt_file(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Assuming the .rpt file has a header and is tab-separated
    data = [line.strip().split('\t') for line in data]
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)
    return df

def replace_null_with_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all occurrences of the string 'NULL' with None in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with string 'NULL' replaced by None.
    """
    return df.replace('NULL', None)



def split_data(df: pd.DataFrame, target_column: list) -> Data:
    """
    Split the DataFrame into dependent and independentdata.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (list): List of target column names

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The dependent and independent data and labels.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]
    return Data(X=X, y=y)

def remove_rows_with_null_values(df: pd.DataFrame, subset_columns: list) -> pd.DataFrame:
    """
    Iterate though subset_columns and remove rows if there is any null value in the subset_columns

    Parameters:
        df: data
        subset_columns: columns to check for null values

    Returns:
    """
    for col in subset_columns:
        df = df[df[col].notnull()]

    return df

def impute_missing_power_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing power values with the mean power value of each cylinder group.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with missing power values imputed.
    """
    mean = df[["Power", "Cylinders"]].groupby(["Cylinders"]).mean()
    df["Power"] = df.apply(
        lambda row: mean.loc[row["Cylinders"]]["Power"] if pd.isnull(row["Power"]) else row["Power"], axis=1
    )
    return df

  
def get_correct_dtype(df: pd.DataFrame, numeric_vars: list, categorical_vars: list, date_vars: list) -> pd.DataFrame:  
    """  
    Corrects the data types of variables in the DataFrame based on specified lists of numeric, categorical, and date variables.  
  
    Parameters:  
        df (pd.DataFrame): The input DataFrame to be modified.  
        numeric_vars (list): List of column names that should be converted to numeric types.  
        categorical_vars (list): List of column names that should be converted to categorical types.  
        date_vars (list): List of column names that should be converted to datetime types.  
  
    Returns:  
        pd.DataFrame: The DataFrame with corrected data types.  
    """  
      
    # Convert specified columns to numeric type  
    for col in numeric_vars:  
        if col in df.columns.tolist():
            df[col] = pd.to_numeric(df[col], errors='coerce')  
      
    # Convert specified columns to categorical type  
    for col in categorical_vars:
        if col in df.columns.tolist():
            df[col] = df[col].astype('category')  
      
    # Convert specified columns to datetime type  
    for col in date_vars:  
        if col in df.columns.tolist():
            df[col] = pd.to_datetime(df[col], errors='coerce')  
      
    return df  


def get_available_free_service_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    To obtain the remaining free service km after deducting the current km from the total km.
    Args:
        df: data

    Returns:

    """
    df['WarrantyKM'] = df['WarrantyKM'].fillna(0)
    df['AvailableWarrantyKM'] = (df['WarrantyKM'] - df['KM']).clip(lower=0)
    return df

def get_available_free_warranty_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    To obtain the free warranty years remaining after deducting the age of the car in months from the total warranty years.
    Args:
        df: data

    Returns:

    """

    df['WarrantyYears'] = df['WarrantyYears'].fillna(0)
    df['AvailableWarrantyYears'] = (df['WarrantyYears'] - df['Age_Comp_Months']).clip(lower=0)
    return df