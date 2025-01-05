from core_model import PricePredictionModel
import logging
from config import ModelVariables


if __name__ == '__main__':
    model_name = "PricePredictionModel"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=f"./logs/{model_name}.log", filemode='a')
    logger = logging.getLogger(model_name)

    model = PricePredictionModel(logger=logger, config=ModelVariables())
    model.run_model()
