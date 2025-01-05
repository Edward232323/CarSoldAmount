import pandas as pd
from config import ModelVariables
import prediction_utils
from data_utils import read_rpt_file, get_correct_dtype, get_available_free_warranty_year, get_available_free_service_km, replace_null_with_none, remove_rows_with_null_values, split_data, impute_missing_power_value
from logging import Logger
import mlflow
import xgboost as xgb

mlflow.set_tracking_uri('http://localhost:5000')

class PricePredictionModel:
    def __init__(self, config: ModelVariables, logger: Logger):
        self.responding_vars = [config.responding_variable]
        self.data_vars = config.numeric_variables + config.categorical_variables + self.responding_vars + config.date_variables
        self.prediction_vars = list(
            set(self.data_vars + config.engineered_variables) - set(config.excluded_variables) - set(
                self.responding_vars)
        )
        self.logger = logger
        self.logger.info("Initializing model")

    def load_data(self, data_path: str):
        self.logger.info("Loading data from {}".format(data_path))
        data = read_rpt_file(data_path)
        return data

    def transform_data(self, df: pd.DataFrame):
        self.logger.info("Transforming data")
        df_filt = df[self.data_vars]
        df_filt = get_correct_dtype(df=df_filt, numeric_vars=ModelVariables().numeric_variables + self.responding_vars, categorical_vars=ModelVariables().categorical_variables, date_vars=ModelVariables().date_variables)
        df_filt = replace_null_with_none(df_filt)
        df_filt = remove_rows_with_null_values(df_filt, subset_columns=["KM", "Sold_Amount"])

        # Feature Engineering - 3 columns : AvailableWarrantyKM, AvailableWarrantyYears, Year_Sold
        df_filt = get_available_free_warranty_year(df_filt)
        df_filt = get_available_free_service_km(df_filt)
        df_filt['Sold_Year'] = df_filt['Sold_Date'].dt.year
        df_processed = df_filt[self.prediction_vars + self.responding_vars]

        # data = pd.get_dummies(df_processed, columns=ModelVariables().categorical_variables)
        data = prediction_utils.encode_categorical_columns(df_processed)
        data = impute_missing_power_value(data)

        # Remove rows with null values since it is very small percentage (less than 0.1%)
        data = data.dropna()

        return data

    def run_model(self):
        """
        Act as the pipeline to run the model

        :return:
        """
        df_train = self.load_data(data_path='./data/DatiumTrain.rpt')
        df_test = self.load_data(data_path='./data/DatiumTest.rpt')

        df_train_transformed = self.transform_data(df_train)
        df_test_transformed = self.transform_data(df_test)

        data_test = split_data(df_test_transformed, self.responding_vars)
        data_train = split_data(df_train_transformed, self.responding_vars)

        with mlflow.start_run():
            # Set XGBoost parameters
            params = {
                "objective": "reg:squarederror",
                "colsample_bytree": 0.3,
                "learning_rate": 0.1,
                "max_depth": 5,
                "alpha": 0.1,
                "n_estimators": 100
            }
            # Log parameters
            mlflow.log_params(params)

            # Set up
            model = xgb.XGBRegressor(**params)
            model = model.fit(X=data_train.X, y=data_train.y)
            y_predict = model.predict(data_test.X)


            # Log model
            mlflow.xgboost.log_model(model, "xgboost_model")
            evaluation_metrics = prediction_utils.evaluate(y_predict, data_test.y['Sold_Amount'])
            mlflow.log_metric("mae", evaluation_metrics.mae)
            mlflow.log_metric("rmse", evaluation_metrics.rmse)

        self.logger.info("Model run completed")


if __name__ == '__main__':
    model_name = "PricePredictionModel"
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=f"./logs/{model_name}.log", filemode='a')
    logger = logging.getLogger(model_name)

    model = PricePredictionModel(logger=logger, config=ModelVariables())
    model.run_model()