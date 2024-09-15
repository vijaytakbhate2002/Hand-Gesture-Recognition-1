import prediction_pipeline as pp
import numpy as np
import cv2 
from config import config
from processes.class_to_char import getChar
import streamlit as st
from typing import Union

def appendText(text:str, el:str) -> Union[str, None]:
    """Args: text (previously generated text)
        Return: string (text with appended element)
        Descritption: fucntion switch between first and second half as per detection
                        or append new character to previous generated text"""
    if el == 'switch':
        if config.CURRENT_HALF == 'first':
            config.CURRENT_HALF = 'second'
            return text, ''
        else:
            config.CURRENT_HALF = 'first'
            return text, ''

    elif config.CURRENT_HALF == 'first':
        text += el
        return text, el
    else:
        text += el
        return text, el

def getResultFromModel(camera_num:int=0, confirmation_ratio:float=0.7, confirmation_list_len:int=20):
    """Args: camera_num, confirmation_ratio (this parameter helps to finalize prediction out of list of predictions)
                confirmation_list_len (specifieses how many predictions should kept in memory to finalize prediction)
        Yield: {"frame":frame, "text":text, "prediction_result":result}
        Description: reads frame from camera"""
    confirmation_by = confirmation_ratio * confirmation_list_len
    confirmation_lis = [-1] * confirmation_list_len
    cp = cv2.VideoCapture(camera_num)
    text = ""
    while True:
        el = ''
        pred_sucess = False
        success, frame = cp.read()
        if not success:
            break

        result = pp.predictionPipe(image=frame)
        frame = cv2.flip(frame, 1)
        
        if result is not None:
            pred_sucess = True
            x_min, x_max, y_min, y_max = result['co_ordinates']
            predicted_class = np.argmax(result['prediction'][0])
            res = getChar(predicted_class=predicted_class, confirmation_by=confirmation_by)
            if res:
                text, el = appendText(text, res)
                config.CONFERMATION_LIS = [-1] * confirmation_lis 
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(255,0,0), thickness=1)
            frame = cv2.putText(frame, el, (x_min, y_min), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 255), thickness=2)
        yield {"frame":frame, "text":text, "prediction_result":result, "success":pred_sucess, "char":el}

    cp.release()


