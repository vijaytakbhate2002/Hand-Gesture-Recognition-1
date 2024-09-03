import sys
sys.path.append('\\'.join(list(__file__.split("\\"))[:-2]))
from config import config
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

class ModelTrainer:
    input_shape = config.CNN_INPUT_IMAGE_SHAPE
    num_classes = config.LABELS
    optimizer_ = config.OPTIMIZER
    loss_ = config.LOSS
    metrics_ = config.METRICS
    epochs_ = config.EPOCHS

    @classmethod
    def createCnnModel(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())  

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))  
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model
    
    @classmethod
    def compileCnnModel(self):
        model = self.createCnnModel()
        model.compile(
            optimizer = self.optimizer_,
            loss = self.loss_,
            metrics = self.metrics_
        )
        return model

    def build(self, X_train, X_test, y_train, y_test):
        """Args: X_train, X_test, y_train, y_test
            Return: Model (Trained CNN model)
            tensorflow will create a log file for visualization with tensorboard"""
        
        tensorboard_callback = TensorBoard(
        log_dir="logs/train",
        histogram_freq=1,               
        write_graph=True,               
        write_images=True,              
        update_freq='epoch',            
        profile_batch=0                 
        )

        model = self.compileCnnModel()
        model.fit(X_train, y_train, epochs=self.epochs_, validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback])
        
        return model