import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from utilities.data_loaders import glove2dict, view_and_load_data
from utilities.data_preprocessors import rejoin_data, build_co_occ_matrix


# This script aims to augment the already existing pre-trained
# word embeddings online which maybe GloVe, Word2Vec etc, which 
# are generalized word embeddings together with the generated 
# hate_speech_dataset which aims to leverage these existing word
# embeddings to generate new word embeddings for these new words
# in the hate_speech_dataset which may not Exist in the vocabulary
# of these word embeddings themselves
if __name__ == "__main__":
    # load the cleaned dataset
    data_path = './data/hate-speech-data-cleaned.csv'
    df_1, all_words, all_unique_words, all_unique_words_counts = view_and_load_data(data_path)

    # here all tokens/words are joined to form a list of all
    # the joined words or the sentences themselves which on
    # the whole is the document
    df_2 = rejoin_data(df_1)
    
    # Get all words not occuring in the pre-trained word embeddings
    # in this important phase we will have to get all words not 
    # occuring in the dictionary we have of the words and their 
    # already existing embeddings. We also generate an important 
    # matrix called the co-occurence matrix in order to train our
    # word embedding model with the use of the existign weights/embeddings 
    # of GloVes dictionary to unseen words in our hate speech dataset
    pre_glove = glove2dict('./embeddings/glove.42B.300d.txt')

    # get all the words in our current corpus that is not 
    # in our dictionary of words and their respective embeddings
    oov = [token for token in all_unique_words if token not in pre_glove.keys()]
    oov_vocab = list(set(oov))
    print(f'list of words not in glove: \n{oov_vocab}\n')
    print(f'length of OOV words: {len(oov_vocab)}\n')
    
    # build co-occurence matrix of OOV words
    co_occ_matrix = build_co_occ_matrix(oov_vocab, df_2['comment'])
    print(f'the co-occurence matrix: \n{co_occ_matrix}\n')
    print(f'shape of the co-occurence matrix: {co_occ_matrix.shape}\n')

    # save co-occurence matrix to .txt file
    np.savetxt('./embeddings/hate_co_occ_matrix.txt', co_occ_matrix)

    
    