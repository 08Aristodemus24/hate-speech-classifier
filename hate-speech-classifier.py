import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import re
import pandas
       

df = pandas.read_csv('./data/hate-offensive-speech.csv',  index_col=0)[['class', 'tweet']]
print(df)
print(df['tweet'][0])
ps = PorterStemmer()
wordnet = WordNetLemmatizer()



# remove non-alphabetic characters
# remove numbers
# remove 
# keep:
# <char>.<space>
# <char>'<char> punctuation
# <char>*<char> that denote cuss words 
df['tweet'] = df['tweet'].apply(lambda tweet: re.sub(r"[^a-zA-Z*][^(\w+'(*)\w+)]", ' ', tweet))
df['tweet'] = df['tweet'].apply(lambda tweet: re.sub(r"[\s]{2,10}", ' ', tweet))
print(df['tweet'][0])

# split all words based on spacing and lower all words in each tweet
df['tweet'] = df['tweet'].apply(lambda tweet: tweet.lower().split(' '))
print(df['tweet'][0])    
print(df['tweet'])
# final = []
# for sentence in df['tweet']:
#     temp = sentence.lower()
    
#     # extract words in sentence
#     temp = sentence.split(' ')
    
#     # remove stop words
#     sw_set = stopwords.words('english')
    
#     # check if word is in stopwords if it is do not add to list
#     # of words
#     # lemmatize word as well
#     temp = [wordnet.lemmatize(word) for word in temp if not word in sw_set]
    
#     # rejoin all non stop words
#     temp = " ".join(temp)
#     final.append(temp)

# print(final)

# tf_idf_m = v.fit_transform(final).toarray()

# data = {}
# for index, sent_vec in enumerate(tf_idf_m):
#     data[final[index]] = sent_vec

# print("final matrix: ", data, end='\n\n')
# print("shape: {}".format(tf_idf_m.shape))







                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          



