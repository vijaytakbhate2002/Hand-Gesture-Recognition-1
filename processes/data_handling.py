import numpy as np
from config import config
import cv2
import os

def trainImagesReader() -> dict:
    """Args: None
        Return: None (Reads all images from train folder and return one dictionary)"""
    train_images = {}
    for folder in os.listdir(config.TRAIN_DATA_PATH):
        train_images[folder] = []
        folder_path = os.path.join(config.TRAIN_DATA_PATH, folder)
        print(f"Reading from {folder_path}...")
        for img_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_name))
            img = cv2.resize(img, config.PROCESSED_IMAGE_SHAPE)
            train_images[folder].append(img)
        print("Done")
    return train_images

def testImagesReader() -> dict:
    """Args: None
        Return: None (Reads all images from train folder and return one dictionary)"""
    test_images = {}
    for folder in os.listdir(config.TEST_DATA_PATH):
        test_images[folder] = []
        folder_path = os.path.join(config.TEST_DATA_PATH, folder)
        print(f"Reading from {folder_path}...")
        for img_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_name))
            img = cv2.resize(img, config.PROCESSED_IMAGE_SHAPE)
            test_images[folder].append(img)
        print("Done")
    return test_images