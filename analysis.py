import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def show_stats(y_true, y_pred):
    # convert one-hot / softmax to class labels list
    y_true = [np.argmax(y_i) for y_i in y_true]
    y_pred = [np.argmax(y_i) for y_i in y_pred]

    print("Confusion Matrix :")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report :")
    print(classification_report(y_true, y_pred, digits=4))
    print('Weighted FScore: \n ', precision_recall_fscore_support(y_true, y_pred, average='weighted'))