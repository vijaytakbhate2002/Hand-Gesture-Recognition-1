from datetime import datetime
from image_processing import data_handling, model_training, model_handling, data_splitter

def trainPipe() -> None:
    data = data_handling.testImagesReader()
    X_train, X_test, y_train, y_test = data_splitter.dataSplitter(data)

    trainer = model_training.ModelTrainer()
    model = trainer.build(X_train, X_test, y_train, y_test)
    
    current_datetime = datetime.today()
    datetime_string = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
    model_handling.modelDumper(datetime_string + '.keras', model=model)

