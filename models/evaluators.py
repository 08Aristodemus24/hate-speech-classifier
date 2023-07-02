import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
from keras import backend as K

import sys

def accuracy(model, X, Y):
    # number of instances of input X
    m = X.shape[0]

    # make predictions
    Y_preds = model.predict(X)

    # calculate how many predictions were right
    acc = np.sum((Y_preds == Y) / m)
    results = 'Accuracy: {:.2%}'.format(acc)

    return acc, results



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



# this is the script where both produced baseline and tuned modesl by model_trainining notebook can be tested
if __name__ == "__main__":
    if sys.argv[1].lower() == "baseline":
        baseline_model = tf.keras.models.load_model('./trained_models/baseline_model.h5')
        
        # evaluate on test dataset the trained model
        # print loss, binary cross entropy cost, and accuracy
        results = baseline_model.evaluate(X_tests, Y_tests)
        print(results)

        res_acc = accuracy(baseline_model, X_tests, Y_tests)
        print(res_acc[1])

        # make predictions and see confusion matrix
        logits = baseline_model.predict(X_tests)
        Y_preds = (tf.nn.sigmoid(logits).numpy() >= 0.5).astype(int)
        conf_matrix = confusion_matrix(Y_tests, Y_preds)
        sb.heatmap(conf_matrix, annot=True)
        plt.show()

    else:
        tuned_model = tf.keras.models.load_model('./trained_models/tuned_model.h5')

        results = tuned_model.evaluate()


    