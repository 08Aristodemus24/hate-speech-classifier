import pandas as pd
import numpy as np
from utilities.data_preprocessors import my_clean

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
    data = pd.read_csv("./ethos_data/Ethos_Dataset_Multi_Label.csv", delimiter=';')

    XT = data['comment'].values
    X = []

    # Add all the labels here 
    yT = data.loc[:, data.columns != 'comment'].values
    y = []

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
