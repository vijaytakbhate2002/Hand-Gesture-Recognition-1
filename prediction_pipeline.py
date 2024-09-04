from processes import image_processing
import numpy as np
import matplotlib.pyplot as plt

recognizer = image_processing.HandLandmark()

def predictionPipe(image) -> dict:
    """Args: image (image from camera or any unprocessed image)
        Return: int (model prediction)"""
    recognizer.fit(image)
    result = recognizer.frame_process()
    if result['flag']:
        prediction = recognizer.predict(image=result['cropped_image'])
        result['prediction'] = prediction
    else:
        result = None
    return result 
