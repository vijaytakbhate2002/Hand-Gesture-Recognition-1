from tensorflow import keras
from config import config
from typing import Union

def modelLoader(model_name:str) -> Union[None, keras.Model]:
    """Args: str (Model path to read)
        Return: model"""
    try:
        model = keras.models.load_model(model_name)
        return model
    except Exception as e:
        raise e

def modelDumper(model_name:str, model) -> None:
    """Args: str (Model path to read)
        Return: None"""
    
    if ".keras" not in model_name:
        raise ValueError("model_path should end with .keras")
    else:
        try:
            model.save(config.MODEL_SAVE + "\\" + model_name)
        except Exception as e:
            raise e

