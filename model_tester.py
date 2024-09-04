import os
from tensorflow import keras
from config import config
import numpy as np
from processes import model_handling, image_processing, data_handling
from processes.image_processing import HandLandmark

def testModel(model_name:str, image) -> list:
    model = model_handling.modelLoader(model_name=model_name)
    test_images = data_handling.testImagesReader()
    test_images = test_images/255.0
    result = model.predict([test_images])
    result = [np.argmax(val) for val in result]
    return result


