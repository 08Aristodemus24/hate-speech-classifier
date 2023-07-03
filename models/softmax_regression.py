import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class SoftmaxRegression:
    def __init__(self, 
                 X, 
                 Y, 
                 epochs=5000,
                 rec_ep_at=500,
                 learning_rate=0.01,  
                 regularization='L2',
                 _lambda=0.1) -> None:
        self.X = X
        self.Y = Y
        self.epochs = epochs

        # X will be (num_features x num_instances)
        self.num_features = X.shape[0]
        self.num_instances = X.shape[1]

        # Y will be (num_classes x num_instances)
        self.num_classes = Y.shape[0]
        self.learning_rate = learning_rate
        self.cost_per_iter = []

        self._THETA = np.nan
        self._BETA = np.nan
            

    def linear(self, X, W, B):
        return tf.linalg.matmul(W, X) + B 
    

    def train(self):
        X = tf.constant(self.X)
        Y = tf.constant(self.Y)

        # initializer parameters e.g. because X is (nf x m) and Y is
        # (nc x m) theta is (nc x nf) and beta is (nc x 1)
        THETA = tf.Variable(tf.random.normal(shape=(self.num_classes, self.num_features), dtype=tf.float64))
        BETA = tf.Variable(tf.zeros(shape=(self.num_classes, 1)), dtype=tf.float64)

        # initialize optimizer
        optimizer = Adam(learning_rate=self.learning_rate)

        # run training
        for epoch in range(self.epochs):
            with GradientTape() as tape:
                # calculate logits or the linear activation of the weights and input
                Z = self.linear(X, THETA, BETA)
                
                # calculate the softmax as well as the cross entropy for each logit
                # by passing the logits Z and the real Y output labels
                cost = self.J_cross_entropy(Z, Y)

                # pass A to cross entropy function
                
            # derivative of cost with respect to THETA and BETA
            grads = tape.gradient(cost, [THETA, BETA])

            # apply gradients to
            optimizer.apply_gradients(zip(grads, [THETA, BETA]))

            print(f'cost at epoch {epoch}: {cost}\n')

        # set new coefficients after training
        self.THETA = THETA
        self.BETA = BETA

    @property
    def THETA(self):
        return self._THETA
    
    @property
    def BETA(self):
        return self._BETA
    
    @THETA.setter
    def THETA(self, new_theta):
        self._THETA = new_theta

    @BETA.setter
    def BETA(self, new_beta):
        self._BETA = new_beta

    def J_cross_entropy(self, Z, Y):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Z, Y))
        return cross_entropy_loss
    
    def predict(self, X_test):
        # retrieve newly trained theta and beta coefficients
        THETA = self.THETA
        BETA = self.BETA

        # calculate logits or Z again but this time with the trained
        # coefficients and bias-coefficients
        logits = tf.linalg.matmul(BETA, X_test) + THETA
        probabilities = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probabilities, axis=1)
        Y_pred = tf.one_hot(predicted_labels, depth=self.num_classes)

        return Y_pred