import os
import numpy as np

# Required paths
ROOT_PATH = '\\'.join(list(__file__.split('\\'))[:-2])
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

# Models
GOOGLE_MODEL = os.path.join(ROOT_PATH, "google_mediapipe_models\\gesture_recognizer.task")
DOCUMENTATION = os.path.join(ROOT_PATH, "documentation\\code_nomenclature.docx")
JSON_PATH = os.path.join(ROOT_PATH, 'generated_text.json')

# Processing constants
PROCESSED_IMAGE_SHAPE = (40,40)
CNN_INPUT_IMAGE_SHAPE = (40, 40, 3)  

# Methods of processing 
NORMALIZATION = "MinMaxScaler"

# Compilation parameters
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

# Model save path
SAVED_MODELS = "trained_models"

# Data handling
TRAIN_CLASSES = os.listdir(TRAIN_DATA_PATH)
TEST_CLASSES = os.listdir(TEST_DATA_PATH)

# Model parameters
LABELS = len(TRAIN_CLASSES)
TEST_SIZE = 0.3
EPOCHS = 10

# current model 
CURRENT_MODEL = "trained_models\\2024-09-01 23-31-08.keras"
CONFIRMATION_OUT_OFF =  20
CONFERMATION_LIS = [-1] * CONFIRMATION_OUT_OFF

SECOND_HALF = {
                0: 'n', 1: 'o', 2: 'p', 3: 'q', 4: 'r', 5: 's', 
               6: 't', 7: 'u', 8: 'v', 9: 'w', 10: 'x', 11: 'y', 
               12: 'z',13: ' ', 14: '-', 15: '?', 16: 'switch'}
FIRST_HALF = {
                0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 
                8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: ' ', 14: '-', 
                15: '?', 16: 'switch'}

SESSION_TEXT = ''''''
SESSION_IMG = np.zeros((500,500,3))
CURRENT_HALF = "first"