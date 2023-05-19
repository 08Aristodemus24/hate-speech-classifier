import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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



