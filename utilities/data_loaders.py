import pandas as pd
import numpy as np
from utilities.data_preprocessors import my_clean
import tqdm
import csv

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


def construct_embedding_matrix(word_emb_path, word_index, EMB_VEC_LEN):
    embedding_dict = {}
    with open(word_emb_path, 'r') as f:
        for line in f:
            # each line consists of: <word> <feature 1> <feature 2> ... <feature d>
            # where d is the 300th feature of the word embedding of that word
            values = line.split()

            # get the word
            word = values[0]

            # if such word exists in our tokenized dictionary
            # then if the reverse is the case that means that 
            # that word exists in the 1.9m word vocab of glove
            if word in word_index.keys():
                # get the vector
                vector = np.asarray(values[1:], 'float32')

                # build the key and value pair of this word and its vector representation
                embedding_dict[word] = vector

    # oov words (out of vacabulary words) will be mapped to 0 vectors
    # this is why we have a plus one always to the number of our words in 
    # our embedding matrix since that is reserved for an unknown or OOV word
    num_words = len(word_index) + 1

    # initialize it to 0
    embedding_matrix=np.zeros((num_words, EMB_VEC_LEN))

    for word, index in tqdm.tqdm(word_index.items()):
        # skip if, if index is already equal to the number of
        # words in our vocab. A break statement if you will
        if index < num_words:
            # if word does not exist in the pretrained word embedding itself
            # then return an empty array
            vect = embedding_dict.get(word, [])

            # if in cases vect is indeed otherwise an empty array due 
            # to the word existing in the pretrained word embeddings
            # then place it in our embedding matrix. Otherwise its index
            # where a word does not exist will stay a row of zeros
            if len(vect) > 0:
                embedding_matrix[index] = vect[:EMB_VEC_LEN]

    return embedding_matrix


def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed