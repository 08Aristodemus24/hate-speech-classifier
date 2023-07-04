import pandas as pd
import numpy as np
import csv
from io import StringIO

from utilities.data_preprocessors import my_clean, read_preprocess, series_to_1D_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(path='', dataset='hate-offensive-speech'):

    if dataset == "hate-offensive-speech":
        df = pd.read_csv(path, index_col=0)
        return df.loc[:, ['tweet', 'class']]
    
    elif dataset == "slur-corpus":
        df = pd.read_csv(path, index_col=0)
        df.index.name = None
        df.reset_index(drop=True, inplace=True)
        return df.loc[:, ['body', 'gold_label']]
    
    elif dataset == "ethos":
        return 0
    
    elif dataset == "reddit":
        return 0 

def load_binary_data(preprocessed=True, stemming_a=True):
    data = pd.read_csv("./data/ethos_data/Ethos_Dataset_Binary.csv", delimiter=';')
    print(data)

    # the comment column
    XT = data['comment'].values

    # the isHate column is the target/real y output value 
    # that tells whether a comment is hate speech or not
    yT = data['isHate'].values

    # some isHate values are floats in between 0 and 1 inclusively
    # encode all greater than or equal 0.5 to 1 and otherwise 0
    print(f'X trains: {XT}\n')
    print(f'Y trains: {yT}\n')

    data['isHate'] = (data['isHate'] >= 0.5).astype('int')
    
    if preprocessed:
        data['comment'] = data['comment'].apply(lambda comment: my_clean(text=str(comment), stops=False, stemming=stemming_a))

    return data

def load_multi_label_data(preprocessed=True, stemming_a=True):
    data = pd.read_csv("./data/ethos_data/Ethos_Dataset_Multi_Label.csv", delimiter=';')

    XT = data['comment'].values
    X = []

    # Add all the labels here 
    yT = data.loc[:, data.columns != 'comment'].values
    y = []

    print(f'X trains: {XT}\n')
    print(f'Y trains: {yT}\n')

    for yt in yT:
        yi = []
        for i in yt:
            if i >= 0.5:
                yi.append(int(1))
            else:
                yi.append(int(0))
        y.append(yi)
        
    for x in XT:
        if preprocessed:
            X.append(my_clean(text=str(x), stops=False, stemming=stemming_a))
        else:
            X.append(x)
    return np.array(X), np.array(yT), np.array(y)



def glove2dict(glove_filename):
    """
    loads the pre-trained word embeddings by GloVe
    """

    with open(glove_filename, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed



def load_co_occ_matrix(path):
    """
    loads the co-occurence matrix built by build_new_corpus.py
    """
    with open(path) as file:
        matrix = file.read()
        matrix = StringIO(matrix)
        
        file.close()

        final = np.loadtxt(matrix).astype(np.int64)
    

    return final



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



def data_split_metric_values(Y_true, Y_pred, unique_labels):
    """
    Y_true - a vector of the real Y values of a data split e.g. the 
    training set, validation set, test

    Y_pred - a vector of the predicted Y values of an ML model given 
    a data split e.g. a training set, validation set, test set

    unique_labels - the uniqeu values of the target/real Y output
    values. Note that it is not a good idea to pass the unique labels
    of one data split since it may not contain all unique labels

    given these arguments it creates a bar graph of all the relevant
    metrics in evaluating an ML model e.g. accuracy, precision,
    recall, and f1-score.
    """
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_true, Y_pred)
    print('Accuracy: {:.2%}'.format(accuracy))

    # precision tp / (tp + fp)
    precision = precision_score(Y_true, Y_pred, labels=unique_labels, average='weighted')
    print('Precision: {:.2%}'.format(precision))

    # recall: tp / (tp + fn)
    recall = recall_score(Y_true, Y_pred, labels=unique_labels,average='weighted')
    print('Recall: {:.2%}'.format(recall))

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_true, Y_pred, labels=unique_labels,average='weighted')
    print('F1 score: {:.2%}'.format(f1))

    return accuracy, precision, recall, f1