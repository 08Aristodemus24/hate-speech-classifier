import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def simple_preprocess(text_string):
  # remove whitespaces
  temp = text_string.strip()
  # view_sentence('whitespace removal', temp)

  # match all non-alphabetic characters not being used by 
  # words then remove e.g. you're uses ' so do not remove '
  # temp = re.sub(r'\b\w*([^\w\s]|_)\w*\b|([^\w\s]|_)', lambda match: match.group(0) if match.group(1) else '', temp)
  temp = re.sub(r'[^a-zA-Z\s]*', '', temp)
  

  # turn sentences to lowercase
  temp = temp.lower()
  print(temp)
  # view_sentence('to lowercase',temp)

  # tokenize sentences and encode as well in unicode format
  words = temp.split(' ')
  # view_sentence('tokenization', temp)

  # remove stop words
  other_exclusions = ["#ff", "ff", "rt", "amp"]
  stop_words = stopwords.words('english')
  stop_words.extend(other_exclusions)
  
  words = [word for word in words if not word in stop_words]
  # view_sentence('stop word removal', temp)

  # lemmatize or stem words/tokens in each row
  # ps = PorterStemmer()
  wordnet = WordNetLemmatizer()
  words = [wordnet.lemmatize(word) for word in words]
  
  
  # encode to utf-8
  # temp = temp.apply(lambda words: [word.encode('utf-8') for word in words])

  
  return temp

def preprocess(text_string):
    print(text_string)
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


# def encode_targets()

   