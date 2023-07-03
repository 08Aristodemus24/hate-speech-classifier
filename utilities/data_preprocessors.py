import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import ast
import pandas as pd
import tqdm

# for ethos dataset
def my_clean(text, stops=False, stemming=False):
    text = str(text)

    text = re.sub(r" US ", " american ", text)
    text = text.lower()
    
    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"didn't", "did not ", text)
    text = re.sub(r"aren't", "are not ", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"shouldn't", "should not ", text)
    text = re.sub(r"couldn't", "could not ", text)
    text = re.sub(r"mustn't", "must not ", text)
    text = re.sub(r"wouldn't", "would not ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"she's", "she is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"that's", "that is ", text)

    text = re.sub(r"could've", "could have ", text)
    text = re.sub(r"would've", "would have ", text)
    text = re.sub(r"should've", "should have ", text)
    text = re.sub(r"must've", "must have ", text)
    text = re.sub(r"i've", "i have ", text)
    text = re.sub(r"we've", "we have ", text)

    text = re.sub(r"you're", "you are ", text)
    text = re.sub(r"they're", "they are ", text)
    text = re.sub(r"we're", "we are ", text)

    text = re.sub(r"you'd", "you would ", text)
    text = re.sub(r"they'd", "they would ", text)
    text = re.sub(r"she'd", "she would ", text)
    text = re.sub(r"he'd", "he would ", text)
    text = re.sub(r"it'd", "it would ", text)
    text = re.sub(r"we'd", "we would ", text)

    text = re.sub(r"you'll", "you will ", text)
    text = re.sub(r"they'll", "they will ", text)
    text = re.sub(r"she'll", "she will ", text)
    text = re.sub(r"he'll", "he will ", text)
    text = re.sub(r"it'll", "it will ", text)
    text = re.sub(r"we'll", "we will ", text)

    text = re.sub(r"\n't", " not ", text) #
    text = re.sub(r"\'s", " ", text) 
    text = re.sub(r"\'ve", " have ", text) #
    text = re.sub(r"\'re", " are ", text) #
    text = re.sub(r"\'d", " would ", text) #
    text = re.sub(r"\'ll", " will ", text) # 
    
    text = re.sub(r"%", " percent ", text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    text = [w for w in text if len(w) >= 2]
    if stemming and stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stemming:
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    
    return text



def simple_preprocess(text_string):
    # turn sentences to lowercase
    temp = text_string.lower()
    # view_sentence('to lowercase',temp)

    # following substitutions are for words with contractions e.g. don't -> do not
    temp = re.sub(r"don't", "do not ", temp)
    temp = re.sub(r"didn't", "did not ", temp)
    temp = re.sub(r"aren't", "are not ", temp)
    temp = re.sub(r"weren't", "were not", temp)
    temp = re.sub(r"isn't", "is not ", temp)
    temp = re.sub(r"can't", "cannot ", temp)
    temp = re.sub(r"doesn't", "does not ", temp)
    temp = re.sub(r"shouldn't", "should not ", temp)
    temp = re.sub(r"couldn't", "could not ", temp)
    temp = re.sub(r"mustn't", "must not ", temp)
    temp = re.sub(r"wouldn't", "would not ", temp)

    temp = re.sub(r"what's", "what is ", temp)
    temp = re.sub(r"that's", "that is ", temp)
    temp = re.sub(r"he's", "he is ", temp)
    temp = re.sub(r"she's", "she is ", temp)
    temp = re.sub(r"it's", "it is ", temp)
    temp = re.sub(r"that's", "that is ", temp)

    temp = re.sub(r"could've", "could have ", temp)
    temp = re.sub(r"would've", "would have ", temp)
    temp = re.sub(r"should've", "should have ", temp)
    temp = re.sub(r"must've", "must have ", temp)
    temp = re.sub(r"i've", "i have ", temp)
    temp = re.sub(r"we've", "we have ", temp)

    temp = re.sub(r"you're", "you are ", temp)
    temp = re.sub(r"they're", "they are ", temp)
    temp = re.sub(r"we're", "we are ", temp)

    temp = re.sub(r"you'd", "you would ", temp)
    temp = re.sub(r"they'd", "they would ", temp)
    temp = re.sub(r"she'd", "she would ", temp)
    temp = re.sub(r"he'd", "he would ", temp)
    temp = re.sub(r"it'd", "it would ", temp)
    temp = re.sub(r"we'd", "we would ", temp)

    temp = re.sub(r"you'll", "you will ", temp)
    temp = re.sub(r"they'll", "they will ", temp)
    temp = re.sub(r"she'll", "she will ", temp)
    temp = re.sub(r"he'll", "he will ", temp)
    temp = re.sub(r"it'll", "it will ", temp)
    temp = re.sub(r"we'll", "we will ", temp)

    temp = re.sub(r"\n't", " not ", temp) #
    temp = re.sub(r"\'s", " ", temp) 
    temp = re.sub(r"\'ve", " have ", temp) #
    temp = re.sub(r"\'re", " are ", temp) #
    temp = re.sub(r"\'d", " would ", temp) #
    temp = re.sub(r"\'ll", " will ", temp) # 
    
    temp = re.sub(r"i'm", "i am ", temp)
    temp = re.sub(r"%", " percent ", temp)

    # match all non-alphabetic characters not being used by 
    # words then remove e.g. you're uses ' so do not remove '
    # temp = re.sub(r'\b\w*([^\w\s]|_)\w*\b|([^\w\s]|_)', lambda match: match.group(0) if match.group(1) else '', temp)
    temp = re.sub(r'[^a-zA-Z\s]*', '', temp)

    # remove whitespaces
    temp = temp.strip()
    print(temp)
    # view_sentence('whitespace removal', temp)

    # tokenize sentences and encode as well in unicode format
    words = temp.split(' ')
    # view_sentence('tokenization', temp)

    # remove stop words
    other_exclusions = ["#ff", "ff", "rt", "amp", ""]
    stop_words = stopwords.words('english')
    stop_words.extend(other_exclusions)

    words = [word for word in words if not word in stop_words]
    # view_sentence('stop word removal', temp)

    # lemmatize or stem words/tokens in each row
    # ps = PorterStemmer()
    wordnet = WordNetLemmatizer()
    words = [wordnet.lemmatize(word) for word in words]

    return words



def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """

    space_pattern = '\s+'

    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    mention_regex = '@[\w\-]+'

    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    
    return parsed_text



def decode_targets(label):
    """
    Y - is a "vector" of shape m x 1 with encoded categorical labels 0 to 4 inclusive
    """ 

    if label == 0:
        return 'APR'
    
    elif label == 1:
        return 'NDG'
    
    elif label == 2:
        return 'DEG'



# how do I convert where for example [[0, 1, 2, 3]] is converted to [[APR, NDG, DEG, HOM]]
def re_encode_tweet_targets(label):
    if label == 0:
        # re-encode 0 which is hate tweet to 2 which is derogatory
        return 2
    elif label == 1:
        # re-encode 1 which is offensive tweet to 0 which is appropriative
        return 0
    elif label == 2:
        # re-encode 2 which is neither off. or hate to 1 which is non-derogatory
        return 1



def re_encode_ethos_targets(label):
    if label == 1:
        # re-encode 1 which is hate comment to 2 which is derogatory/hate
        return 2
    elif label == 0:
        # re-encode 0 which is non hate comment to which is non-derogatory
        return 1
    


def read_preprocess(df):
    """preprocess data
    by converting string to list or series
    """

    # saving df to csv does not preserve a series type and is 
    # converted to str so convert first str to list or series
    df['comment'] = df['comment'].apply(lambda comment: ast.literal_eval(comment))

    return df



def series_to_1D_array(series):
    """this converts the series or column of a df
    of lists 
    """

    return [item for sublist in series for item in sublist]



def construct_embedding_dict(word_emb_path, word_to_index):
    """
    returns an embedding dictionary populated with all the words and
    their respective vector representations from the file of the 
    pretrained embeddings of GloVe 
    """

    embedding_dict = {}
    with open(word_emb_path, 'r') as file:
        for index, line in enumerate(file):
            # each line consists of: <word> <feature 1> <feature 2> ... <feature d>
            # where d is the 300th feature of the word embedding of that word
            values = line.split()

            # get the word
            word = values[0]

            # if such word exists in our tokenized dictionary
            # then if the reverse is the case that means that 
            # that word exists in the 1.9m word vocab of glove
            if word in word_to_index.keys():
                # get the vector
                vector = np.asarray(values[1:], 'float64')
                
                # build the key and value pair of this word and its vector representation
                embedding_dict[word] = vector
                
                # get embedding vector length which will be constant across all keys
                EMB_VEC_LEN = vector.shape[0]

        file.close()

    return embedding_dict, EMB_VEC_LEN



def construct_embedding_matrix(word_to_index, embedding_dict, EMB_VEC_LEN):
    """
    Constructs the embedding matrix upon finishing the phase of 
    constructing the embedding dictionary. So that reading the GloVe 
    embeddings is only done ones to increase time efficiency
    """

    # oov words (out of vacabulary words) will be mapped to 0 vectors
    # this is why we have a plus one always to the number of our words in 
    # our embedding matrix since that is reserved for an unknown or OOV word
    vocab_len = len(word_to_index) + 1

    # initialize it to 0
    embedding_matrix = np.zeros((vocab_len, EMB_VEC_LEN))

    for word, index in tqdm.tqdm(word_to_index.items()):
        # skip if, if index is already equal to the number of
        # words in our vocab. A break statement if you will
        if index < vocab_len:
            # if word does not exist in the pretrained word embedding itself
            # then return an empty array
            vector = embedding_dict.get(word, [])

            # if in cases vect is indeed otherwise an empty array due 
            # to the word existing in the pretrained word embeddings
            # then place it in our embedding matrix. Otherwise its index
            # where a word does not exist will stay a row of zeros
            if len(vector) > 0:
                embedding_matrix[index] = vector[:EMB_VEC_LEN]

    return embedding_matrix



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



def sentences_to_avgs(sentences, word_to_vec_dict: dict):
    """
    Converts a series object of sentences (string) into a list of words (strings) then 
    extracts the GloVe representation of each word and averages its value into a single 
    vector encoding the meaning of the sentence. Return a 2D numpy array representing 
    all sentences vector representations
    
    Arguments:
    sentences -- a series object of sentences
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about each sentence, numpy-array of shape (m, d), where m is the
    number of sentences in the dataframe and d is the number of dimensions or the length of a word's vector rep
    """
    
    # Initialize the average word vector, should have the same shape
    # as your word vectors. Use `np.zeros` and pass in the argument 
    # of any word's word 2 vec's shape. Get also a valid word contained
    # in the word_to_vec_map.
    any_word = list(word_to_vec_dict.keys())[0]
    m = sentences.shape[0]
    d = word_to_vec_dict[any_word].shape[0]
    avgs = np.zeros(shape=(m, d))

    # in each sentence split them into individual words then take the 
    # average of these occuring words in the word embedding dictionary
    for index, sentence in enumerate(sentences):
        # Initialize/reset count to 0
        count = 0

        # Step 1: Split sentence into list of lower case words 
        words = sentence.lower().split(' ')
        
        # Step 2: average the word vectors. You can loop over the words in the list "words".
        for word in words:
            # Check that word exists in word_to_vec_dict
            if word in word_to_vec_dict:
                
                # add the d-dim vector representation of the word 'word'
                # to the avg variable that will contain our summed 
                # vectors of a sentences words
                avgs[index] += word_to_vec_dict[word]
                
                # Increment count which represents how many words we have in our sentence
                count +=1
            
        if count > 0:
            # Get the average. But only if count > 0
            avgs[index] = avgs[index] / count
    
    return avgs