import numpy as np

from sklearn.metrics import confusion_matrix

def show_stats(y_true, y_pred):
    # convert one-hot / softmax to class labels list
    y_true = [np.argmax(y_i) for y_i in y_true]
    y_pred = [np.argmax(y_i) for y_i in y_pred]

    print(confusion_matrix(y_true, y_pred))