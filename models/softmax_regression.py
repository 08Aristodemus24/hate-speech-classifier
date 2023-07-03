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
                 train_X, 
                 train_Y, 
                 val_X, 
                 val_Y,
                 epochs=5000,
                 rec_ep_at=500,
                 learning_rate=0.03,  
                 regularization='L2',
                 lambda_=0.1,
                 ) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y

        self.epochs = epochs
        self.rec_ep_at = rec_ep_at

        # X will be (num_instances x num_features)
        self.num_instances = train_X.shape[0]
        self.num_features = train_X.shape[1]
        
        # Y will be (num_classes x num_instances)
        self.num_classes = train_Y.shape[1]

        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.regularization = regularization

        self.cost_per_iter = []

        self._THETA = np.nan
        self._BETA = np.nan


    def view_vars(self, X, Y, THETA, BETA):
        print(f'X: {X}\n')
        print(f'Y: {Y}\n')
        print(f'THETA: {THETA}\n')
        print(f'BETA: {BETA}\n')

            
    def linear(self, X, W, B):
        return tf.linalg.matmul(X, W) + B
    

    def train(self, show_vars=True):
        # maybe hte bug occurs because Y and X are transposed versions of each other
        train_X = tf.constant(self.train_X)
        train_Y = tf.constant(self.train_Y)
        val_X = tf.constant(self.val_X)
        val_Y = tf.constant(self.val_Y)

        # initializer parameters e.g. because X is (m x nf) and Y is
        # (m x nc) theta is (nf x nc) and beta is (nc x 1)
        THETA = tf.Variable(tf.random.normal(shape=(self.num_features, self.num_classes), dtype=tf.float64))
        BETA = tf.Variable(tf.zeros(shape=(1, self.num_classes), dtype=tf.float64))

        # initialize optimizer
        optimizer = Adam(learning_rate=self.learning_rate)

        # run training
        for epoch in range(self.epochs):
            with GradientTape() as tape:
                # calculate logits or the linear activation of the weights and input
                train_Z = self.linear(train_X, THETA, BETA)
                train_A = tf.nn.softmax(train_Z)

                val_Z = self.linear(val_X, THETA, BETA)
                val_A = tf.nn.softmax(val_Z)
                
                # calculate the softmax as well as the cross entropy for each logit
                # by passing the logits Z and the real Y output labels
                # pass A to cross entropy function
                train_cost = self.J_cross_entropy(train_A, train_Y) + self.regularizer(THETA)
                val_cost = self.J_cross_entropy(val_A, val_Y)
                
            # derivative of cost with respect to THETA and BETA
            grads = tape.gradient(train_cost, [THETA, BETA])

            # apply gradients to
            optimizer.apply_gradients(zip(grads, [THETA, BETA]))
            if ((epoch % self.rec_ep_at) == 0) or (epoch == self.epochs - 1):
                print(f'training cost at epoch {epoch}: {train_cost} \t validation cost at epoch {epoch}: {val_cost}\n')

                if show_vars == True:
                    self.view_vars(train_X, train_Y, THETA, BETA)

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

    def J_cross_entropy(self, A, Y):
        # Clip prediction values to avoid log(0) error.
        A = tf.clip_by_value(A, 1e-9, 1.)

        # perform cross entropy calculation where reduce_sum sums
        # all multiplied predictions to its respective y value 
        cross_entropy_cost = tf.reduce_mean(-1 * tf.reduce_sum(Y * tf.math.log(A)))

        # return cost
        return cross_entropy_cost
    
    def regularizer(self, THETA):
        if self.regularization.upper() == "L2":
            # get the square of all coefficients in each layer excluding biases
            # if 5 layers then loop from 0 to 3 to access all coefficients
            sum_sq_coeffs = tf.reduce_sum(tf.math.square(THETA))

            # multiply by lambda constant then divide by 2m
            l2_norm = (self.lambda_ * sum_sq_coeffs) / (2 * self.num_instances)

            # return l2 norm
            return l2_norm

        elif self.regularization.upper() == "L1":
            # if there is only 2 layers then calculation
            # in loop only runs once
            sum_abs_coeffs = tf.reduce_sum(tf.math.abs(THETA))

            # multiply by lambda constant then divide by 2m
            l1_norm = (self.lambda_ * sum_abs_coeffs) / (2 * self.num_instances)

            # return l1 norm
            return l1_norm
    
    def predict(self, X_test):
        # retrieve newly trained theta and beta coefficients
        THETA = self.THETA
        BETA = self.BETA

        # calculate logits or Z again but this time with the trained
        # coefficients and bias-coefficients
        Z = tf.linalg.matmul(X_test, THETA) + BETA
        probabilities = tf.nn.softmax(Z)
        A = tf.argmax(probabilities, axis=1)
        Y_pred = tf.one_hot(A, depth=self.num_classes)

        return Y_pred