import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def simple_preprocess(text_string):
    # match all non-alphabetic characters not being used by 
    # words then remove e.g. you're uses ' so do not remove '
    # temp = re.sub(r'\b\w*([^\w\s]|_)\w*\b|([^\w\s]|_)', lambda match: match.group(0) if match.group(1) else '', temp)
    temp = re.sub(r'[^a-zA-Z\s]*', '', text_string)

    # turn sentences to lowercase
    temp = temp.lower()
    # view_sentence('to lowercase',temp)

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



def decode_targets(Y):
    """
    Y - is a "vector" of shape m x 1 with encoded categorical labels 0 to 4 inclusive
    """ 

# how do I convert where for example [[0, 1, 2, 3]] is converted to [[APR, NDG, DEG, HOM]]

def re_encode_targets(label):
    if label == 0:
        # re-encode 0 which is hate tweet to 2 which is derogatory
        return 2
    elif label == 1:
        # re-encode 1 which is offensive tweet to 0 which is appropriative
        return 0
    elif label == 2:
        # re-encode 2 which is a neither off. or hate to 4 which is non-derogatory
        return 4



