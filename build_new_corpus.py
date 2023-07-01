import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from utilities.data_loaders import glove2dict
from utilities.data_preprocessors import read_preprocess, series_to_1D_array



def view_and_load_data(data_path):
    """
    Load the data:
    * after loadign the data see also the number of all unique words in the dataframe

    * see all the words that occur without using the constraint of the words having to be unique

    * see the unique words themselves
    """

    df_1 = pd.read_csv(data_path, index_col=0)
    df_1 = read_preprocess(df_1)

    all_words = pd.Series(series_to_1D_array(df_1['comment']))
    all_unique_words_counts = all_words.value_counts()
    all_unique_words = all_words.unique()

    print(f'length of all words: {len(all_words)}\n')
    print(f'length of all unique words: {len(all_unique_words)}\n')
    print(f'all unique words: \n{all_unique_words}\n')
    print(f'frequency of all unique words: \n{all_unique_words_counts}\n')

    return df_1, all_words, all_unique_words, all_unique_words_counts


def rejoin_data(df_2):
    """
    Getting important variables:
    * get the list in the dataframe with the greatest amount of words or 
    with the longest sequence, this will be used later for generating
      the embeddings

    * reassign again the df but this time instead of lists of words in the
    comment column join them, this will be again used later for generating 
    the indexed reprsetnations of the sequences of words 

    * before joining again get array in df with longest length first
    """

    df_2['comment'] = df_2['comment'].apply(lambda comment: " ".join(comment))
    sample = df_2.loc[0, 'comment']
    print(f'{sample}')

    return df_2


def build_co_occ_matrix(oov_vocab, document):
    """
    Building the co-occurence matrix:
    * this will build teh co-occurence matrix for all the words not occuring
    in the already pre-trained word and word embedding dictionary available online
    """

    # this will convert the collection of text documents 
    # or sentences to a matrix of token/word counts
    cv = CountVectorizer(ngram_range=(1, 1), vocabulary=oov_vocab)

    # this will create the matrix of token counts
    X = cv.fit_transform(document)

    # matrix multiply X's transpose to X
    Xc = X.T * X

    # set the diagonals to be zeroes as it's pointless to be 1
    Xc.setdiag(0)

    # finally convert Xc to an array once self.setdiag is called
    # this will be our co-occurence matrix to be fed to Mittens
    co_occ_matrix = Xc.toarray()
    
    return co_occ_matrix


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

    
    