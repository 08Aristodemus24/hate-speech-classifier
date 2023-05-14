# %% [markdown]
# ## Access kaggle dataset through generated API key

# %%
# !pip install kaggle
# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# %% [markdown]
# ## download kaggle dataset once api key is loaded in notebook

# %%
# !kaggle datasets download danofer/sarcasm

# %% [markdown]
# ## unzip zip file if there is

# %%
# !unzip sarcasm.zip

# %% [markdown]
# ## Import libraries

# %%
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')


# %% [markdown]
# ## Explore dataset

# %%
df = pd.read_csv('./data/train-balanced-sarcasm.csv')
df

# %%
# Y = pd.read_csv('./test-balanced.csv')
# Y

# %% [markdown]
# ## Split input/independent and output/dependent columns/features

# %%
X = df['parent_comment']
Y = df['label']
print(X.dtype)
print(Y.dtype)

# %%
sample = X.loc[0]
sample = sample.encode('utf-8')
sample = sample.split()
sample

# %% [markdown]
# ## Preprocess text
# - remove trailing whitespaces
# - remove non-alphanumeric characters
# - lower sentences
# - tokenize
# - remove stop words
# - lemmatize or stem word

# %%
import re

def view_sentence(phase, sentences, limit=5):
  for sentence in sentences.iloc[:limit]:
    print(f'{phase} phase:\n{sentence}\n')

def preprocess(comment__df) -> pd.DataFrame:
  # remove whitespaces
  temp = comment__df.apply(lambda sentence: sentence.strip())
  view_sentence('whitespace removal', temp)

  # match all non-alphanumeric characters 
  # not being used by words then remove 
  # e.g. you're uses ' so do not remove '
  temp = temp.apply(lambda sentence: re.sub(r'\b\w*([^\w\s]|_)\w*\b|([^\w\s]|_)', lambda match: match.group(0) if match.group(1) else '', sentence))
  view_sentence('non-alphanumeric char except in words removal', temp, limit=20)

  # turn sentences to lowercase
  temp = temp.apply(lambda sentence: sentence.lower())
  view_sentence('to lowercase',temp)

  # tokenize sentences and encode as well in unicode format
  temp = temp.apply(lambda sentence: sentence.split(' '))
  view_sentence('tokenization', temp)

  # remove stop words
  stop_words = stopwords.words('english')
  print(stop_words)
  temp = temp.apply(lambda words: [word for word in words if not word in stop_words])
  view_sentence('stop word removal', temp)

  # lemmatize or stem words/tokens in each row
  # ps = PorterStemmer()
  wordnet = WordNetLemmatizer()
  temp = temp.apply(lambda words: [wordnet.lemmatize(word) for word in words])
  view_sentence('lemmatization', temp)

  # encode to utf-8
  temp = temp.apply(lambda words: [word.encode('utf-8') for word in words])


  return temp

# %%
X = preprocess(X)

# %% [markdown]
# ## model architecture and initialization

# %% [markdown]
# here the window size or the amount of words to use as context and target are indicated, as well as the min_count which indicates if a word length lower than its value is still to be considered part of the window, and workers which represent the number of threads to use in training the model

# %%
# model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)


# # %% [markdown]
# # ### building the vocabulary

# # %%
# # progress_per arg is the number of words 
# # to proecss before a status update is given
# model.build_vocab(X, progress_per=1000)

# # %%
# # see also corpus count and epochs
# print(model.corpus_count)
# print(model.epochs)
# # print(model.vocab)

# # %% [markdown]
# # ### training the model

# # %%
# model.train(X, total_examples=model.corpus_count, epochs=30)

# # %%
# vocab, vectors = model.wv.key_to_index, model.wv.vectors

# # %%
# vocab

# # %%
# vectors

# # %%
# len(vocab)

# # %%
# vectors.shape

# # %%
# word_vectors = pd.DataFrame({'word': vocab.keys()})
# word_vectors

# # %%
# model.wv.save('./word_embeddings.wordvectors')

# # %%



