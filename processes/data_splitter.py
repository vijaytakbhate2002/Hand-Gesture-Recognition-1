import sys
sys.path.append('\\'.join(list(__file__.split("\\"))[:-2]))
import numpy as np
from sklearn.model_selection import train_test_split

def dataSplitter(data:dict) -> tuple:
    y_labels = [[key] * len(data[key]) for key in data.keys()]
    y_labels = np.concatenate(y_labels) 
    y_labels = np.array([int(num) for num in y_labels])
    print("labels created = ",set(y_labels))
    X = np.concatenate([arr for arr in data.values()])
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels,)
    return X_train, X_test, y_train, y_test
