import tensorflow as tf
from tensorflow import GradientTape
import numpy as np

class SoftmaxRegression:
    def __init__(self, 
                 X, 
                 Y, 
                 epochs=5000,
                 rec_ep_at=500,
                 learning_rate=0.01,  
                 initializer='xavier', 
                 regularization='L1',
                 _lambda=0.1) -> None:
        self.X = X
        self.Y = X

        # initializer parameters e.g. because X is (nf x m) and Y is
        # (nc x m) theta is (nc x nf) and beta is (nc x 1)
        self.THETA = tf.Variable(tf.random.normal(shape=(Y.shape[0], X.shape[0]), dtype=tf.float64))
        self.BETA = tf.Variable(tf.random.normal(shape=(Y.shape[0], 1)), dtype=tf.float64)

        self.epochs = epochs
        self.initializer = initializer

        # X will be (num_features x num_instances)
        self.num_features = X.shape[0]
        self.num_instances = X.shape[1]

        # Y will be (num_classes x num_instances)
        self.num_outputs = Y.shape[0]
        self.learning_rate = learning_rate
        self.cost_per_iter = []


    def softmax(self, Z):
        """Compute softmax values for each sets of scores in x."""
        # e_z = np.exp(z)
        # a = e_z / np.sum(e_z)

        e_Z = np.exp(Z - np.max(Z))
        return e_Z / e_Z.sum()
    
            
    def linear(self, X, W, B):
        return tf.linalg.matmul(W, X) + B 
    

    def train(self):
        for epoch in range(self.epochs):
            with GradientTape() as tape:
                Z = np.dot(self.THETA, )


    def J(self, A, Y):
        loss = np.dot(-1 * Y, np.log(A))
        cost = 0
        return cost

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
    """
    # Get a valid word contained in the word_to_vec_map. 
    # first word here is 'the'
    any_word = list(word_to_vec_map.keys())[0]
    
    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split(' ')

    # Initialize the average word vector, should have the same shape as your word vectors.
    # Use `np.zeros` and pass in the argument of any word's word 2 vec's shape
    avg = np.zeros(word_to_vec_map[any_word].shape)
    
    # Initialize count to 0
    count = 0
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for word in words:
        # Check that word exists in word_to_vec_map
        if word in word_to_vec_map:
            
            # add the 50 dim vector representation fo the word w 
            # to the avg variable that will contain our summed 
            # vectors of a sentences words
            avg += word_to_vec_map[word]
            
            # Increment count which represents how many words we have in our sentence
            count +=1
          
    if count > 0:
        # Get the average. But only if count > 0
        avg = avg / count
    
    ### END CODE HERE ###
    
    return avg



