import numpy as np
from io import StringIO
import gensim
from mittens import GloVe, Mittens


def train_new_embeddings(pre_glove, oov_vocab, co_occ_matrix, dim, epochs):
    """
    Training the Mittens model with the new words
    * since our GloVe word embeddings are basically 300 in length our n 
    arg here should be also 300 we also set max_iter or our number of 
    epochs to train our word embedding model to maybe 1000 to 5000 epochs
    """

    # instantiate the Mittens class
    mittens_model = Mittens(n=dim, max_iter=epochs)

    # this will return only the words not existing in our pre-trained word embeddings
    # but the good thing is we can reshape adn save this file to resemble that of our
    # pretrained word embeddings file
    new_embeddings = mittens_model.fit(
        co_occ_matrix,
        vocab=oov_vocab,
        initial_embedding_dict=pre_glove
    )

    post_glove = dict(zip(oov_vocab, new_embeddings))
    file = open("hate_speech_glove.txt","wb")
    # pickle.dump(post_glove, file)
    file.close()

def load_co_occ_matrix(path):
    with open() as file:
        matrix = file.read()
        matrix = StringIO(matrix)
        
        file.close()

        final = np.loadtxt(matrix).astype(np.int64)
    

    return final



if __name__ == "__main__":
    # load the saved co-occurence matrix
    co_occ_path = "./embeddings/co_occ_matrix_sample.txt"
    co_occ_matrix = load_co_occ_matrix(co_occ_path)
    