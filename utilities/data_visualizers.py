import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import itertools



def view_sentence(sentences, phase='', limit=5):
    for sentence in sentences.iloc[:limit]:
        print(f'{phase}phase:\n{sentence}\n')



def view_words(word_vec: dict, title: str, len_to_show=20, word_range=10):
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

    # for iter, (key, item) in enumerate(sliced_word_vec.items()):
    #     print(f'{key}\n{item[:len_to_show]}')


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
    axis.set_title(title)
    plt.savefig(f'./figures & images/{title}.png')
    plt.show()



def view_word_frequency(word_counts, colormap:str, title: str, kind: str='barh', limit: int=6):
    """
    plots either a horizontal bar graph to display frequency of words top 'limit' 
    words e.g. top 20 or a pie chart to display the percentages of the top 'limit' 
    words e.g. top 20, specified by the argument kind which can be either
    strings barh or pie
    """

    # get either last few words or first feww words
    data = word_counts[:limit].sort_values(ascending=True)
    cmap = cm.get_cmap(colormap)
    _, axis = plt.subplots()
    
    if kind == 'barh':        
        axis.barh(data.index, data.values, color=cmap(np.linspace(0, 1, len(data))))

        # axis = word_counts[:limit].sort_values(ascending=True).plot(kind='barh', colormap='viridis')

        axis.set_xlabel('frequency')
        axis.set_ylabel('words')
        axis.set_title(title)
        plt.savefig(f'./figures & images/{title}.png')

        plt.show()
    elif kind == 'pie':
        axis.pie(data, labels=data.index, autopct='%.2f%%', colors=cmap(np.linspace(0, 1, len(data))))
        axis.axis('equal')
        axis.set_title(title)
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()
    

    
def train_cross_results_v2(results: dict, epochs: list, img_title: str='figure'):
    """
    plots the number of epochs against the cost given cost values across these epochs
    """
    # # use matplotlib backend
    # mpl.use('Agg')

    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [('p:', '#5d42f5'), ('h-', '#fc03a5'), ('o:', '#1e8beb'), ('x--','#1eeb8f'), ('+--', '#0eb802'), ('8-', '#f55600')]

    for index, (key, value) in enumerate(results.items()):
        axis.plot(np.arange(len(epochs)), value, styles[index][0] ,color=styles[index][1], alpha=0.5, label=key)

    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.set_title(img_title)
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')
    plt.show()
    

    # delete figure
    del figure



def multi_class_heatmap(conf_matrix, img_title: str, cmap: str='YlGnBu'):
    axis = sb.heatmap(conf_matrix, cmap=cmap, annot=True, fmt='g')
    axis.set_title(img_title)

    plt.savefig(f'./figures & images/{img_title}.png')



def view_metric_values(df, img_title: str):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values
    """
    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # Create an array with the colors you want to use
    colors = ['#2ac5b9', '#1ca3b6', '#0a557a', '#01363e',]
    sb.set_palette(sb.color_palette(colors))

    # create accuracy, precision, recall, f1-score of training group
    # create accuracy, precision, recall, f1-score of validation group
    # create accuracy, precision, recall, f1-score of testing group
    df_exp = df.melt(id_vars='data_split', var_name='metric', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='metric', ax=axis)
    axis.set_title(img_title)
    axis.set_yscale('log')
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')
    plt.show()



def view_classified_labels(df, img_title: str):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values
    """
    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # Create an array with the colors you want to use
    colors = ['#db7f8e', '#b27392']
    sb.set_palette(sb.color_palette(colors))

    # create accuracy, precision, recall, f1-score of training group
    # create accuracy, precision, recall, f1-score of validation group
    # create accuracy, precision, recall, f1-score of testing group
    df_exp = df.melt(id_vars='data_split', var_name='status', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='status', ax=axis)
    axis.set_title(img_title)
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')
    plt.show()



def view_label_freq(label_freq, img_title: str):
    axis = sb.barplot(x=["DER", "NDG", "OFF", "HOM"], y=label_freq.values, palette="flare")
    axis.set_title(img_title)

    plt.savefig(f'./figures & images/{img_title}.png')
    plt.show()



def view_final_metrics(results: dict, label: str):
    print(f'\n{label}:')
    for key, item in results.items():
        print(f'{key}: {item[-1]}')

        



