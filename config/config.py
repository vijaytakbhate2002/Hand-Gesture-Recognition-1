import os

# Required paths
ROOT_PATH = '\\'.join(list(__file__.split('\\'))[:-2])
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

# Models
GOOGLE_MODEL = os.path.join(ROOT_PATH, "google_mediapipe_models\\gesture_recognizer.task")
DOCUMENTATION = os.path.join(ROOT_PATH, "documentation\\code_nomenclature.docx")

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
MODEL_SAVE = "trained_models"

# Data handling
TRAIN_CLASSES = os.listdir(TRAIN_DATA_PATH)
TEST_CLASSES = os.listdir(TEST_DATA_PATH)

# Model parameters
LABELS = len(TRAIN_CLASSES)
TEST_SIZE = 0.3
EPOCHS = 10