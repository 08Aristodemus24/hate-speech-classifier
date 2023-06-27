import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import itertools


def view_sentence(sentences, phase='', limit=5):
    for sentence in sentences.iloc[:limit]:
        print(f'{phase}phase:\n{sentence}\n')


def view_words(word_vec: dict, len_to_show=20, word_range=10):
    """
    word_vec - key value pairs of the words and respective embeddings

    len_to_show - the limit in which each word vector is only allowed to show

    word range - if false then all words are shown but if a value 
    is given then number words shown are up to that value only
    
    word_range: int | bool=50
    """

    # slice the dictionary to a particular range
    sliced_word_vec = dict(itertools.islice(word_vec.items(), word_range))

    # separate all word keys and their respective 
    # embeddings from each other and place in separate arrays
    words, embeddings = zip(*sliced_word_vec.items())
    words = np.array(words)
    embeddings = np.array(embeddings)

    for iter, (key, item) in enumerate(sliced_word_vec.items()):
        print(f'{key}\n{item[:len_to_show]}')


    # reduce length/dimensions of embeddings from 300 to 2
    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)

    # because there are 21624 words dimensionality of emb_red will go from 21624 x 300 to 21624 x 2
    emb_red = tsne_model.fit_transform(embeddings)


    # populate a new dictionary with new reduced embeddings with 2 dimensions
    word_vec_red = {}
    for index, key in enumerate(words):
        # extract x and ys in emb_red array
        x, y = emb_red[index]

        # populate dictionary with x and y coordinates
        if key not in word_vec_red:
            word_vec_red[key] = (x, y)


    # build and visualize
    fig = plt.figure(figsize=(15, 15))
    axis = fig.add_subplot()

    # plot the points
    axis.scatter(emb_red[:, 0], emb_red[:, 1], c=np.random.randn(emb_red.shape[0]), marker='p',alpha=0.75, cmap='magma')

    # annotate the points
    for iter, (word, coord) in enumerate(word_vec_red.items()):
        x, y = coord
        axis.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_title('t-SNE reduced word embeddings')
    plt.show()

def view_word_frequency(word_counts, label: str, limit: int=6):
    axis = word_counts[:limit].sort_values(ascending=True).plot(kind='barh', color=plt.get_cmap('magma'))
    axis.set_xlabel('frequency')
    axis.set_ylabel('words')
    axis.set_title('word frequency graph')
    plt.show()

        

        



