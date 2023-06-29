# **STILL IN PRODUCTION** 

## references and citations:
- **https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/blob/master/README.md**

# A Benchmark Dataset for Learning to Intervene in Online Hate Speech

In order to encourage strategies of countering online hate speech, we introduce a novel task of generative hate speech intervention along with two fully-labeled datasets collected from Gab and Reddit. Distinct from existing hate speech datasets, our datasets retain their conversational context and introduce human-written intervention responses. Due to our data collecting strategy, all the posts in our datasets are manually labeled as hate or non-hate speech by Mechanical Turk workers, so they can also be used for the hate speech detection task.

There are two CSV files under the data directory: gab.csv and reddit.csv, These datasets provide conversation segments, hate speech labels, as well as intervention responses written by  Mechanical Turk workers.

Two data files have the same data structure:

|Field|Description|
|-------|-------------|
|id|the ids of the post in a conversation segment|
|text|the text of the posts in a conversation segment|
|hate_speech_idx|a list of the indexes of the hateful posts in this conversation|
|response|a list of human-written responses|

Please refer to the paper "A Benchmark Dataset for Learning to Intervene in Online Hate Speech" (EMNLP 2019) for the detailed information about the dataset.

# Data Processing
Other meta data of the Reddit post can be retrieved using Reddit API and the ids of the posts. 

Other meta data of the Gab post can be retrieved from the dataset https://files.pushshift.io/gab/GABPOSTS_2018-10.xz using the ids of the posts.

# License
The dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA. 


- **https://github.com/08Aristodemus24/Ethos-Hate-Speech-Dataset.git**



# ETHOS Hate Speech Dataset
ETHOS: multi-labEl haTe speecH detectiOn dataSet. This repository contains a dataset for hate speech detection on social media platforms, called Ethos. There are two variations of the dataset:
- Ethos_Dataset_Binary.csv[Ethos_Dataset_Binary.csv] contains 998 comments in the dataset alongside with a label about hate speech *presence* or *absence*. 565 of them do not contain hate speech, while the rest of them, 433, contain. 
- Ethos_Dataset_Multi_Label.csv [Ethos_Dataset_Multi_Label.csv] which contains 8 labels for the 433 comments with hate speech content. These labels are *violence* (if it incites (1) or not (0) violence), *directed_vs_general* (if it is directed to a person (1) or a group (0)), and 6 labels about the category of hate speech like, *gender*, *race*, *national_origin*, *disability*, *religion* and *sexual_orientation*.

## Ethos /ˈiːθɒs/ 
is a Greek word meaning “character” that is used to describe the guiding beliefs or ideals that characterize a community, nation, or ideology. The Greeks also used this word to refer to the power of music to influence emotions, behaviors, and even morals.

Please check our older dataset as well: https://intelligence.csd.auth.gr/topics/hate-speech-detection/

## Reference
Please if you use this dataset in your research cite out preprint paper: [ETHOS: a multi-label hate speech detection dataset](https://rdcu.be/cEoQn)
```
@article{mollas_ethos_2022,
	title = {{ETHOS}: a multi-label hate speech detection dataset},
	issn = {2198-6053},
	url = {https://doi.org/10.1007/s40747-021-00608-2},
	doi = {10.1007/s40747-021-00608-2},
	journal = {Complex \& Intelligent Systems},
	author = {Mollas, Ioannis and Chrysopoulou, Zoe and Karlos, Stamatis and Tsoumakas, Grigorios},
	month = jan,
	year = {2022},
}
```

## Contributors on Ethos
Name | Email
--- | ---
Grigorios Tsoumakas | greg@csd.auth.gr
Ioannis Mollas | iamollas@csd.auth.gr
Zoe Chrysopoulou | zoichrys@csd.auth.gr
Stamatis Karlos | stkarlos@csd.auth.gr

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

- **https://github.com/08Aristodemus24/slur-corpus.git**



# Towards a Comprehensive Taxonomy and Large-Scale Annotated Corpus for Online Slur Usage

Corpus repository for Kurrek, J., Saleem, H. M., & Ruths, D. (2020, November). Towards a Comprehensive Taxonomy and Large-Scale Annotated Corpus for Online Slur Usage. In Proceedings of the Fourth Workshop on Online Abuse and Harms (pp. 138-149). You can read it [here](https://www.aclweb.org/anthology/2020.alw-1.17.pdf).

**CONTENT WARNING: This corpus contains content that is racist, transphobic, homophobic, and offensive in many other ways. Please use responsibly.**

## Comment Annotation Metadata

This corpus consists of 40,000 annotated Reddit comments. For each comment the following details are included.

| FIELD        | INFO                                                                                                |
|--------------|-----------------------------------------------------------------------------------------------------|
| id           | STR. ID to the Reddit comment.                                                                      |
| link_id      | STR. ID to the Reddit post the comment was made in.                                                 |
| parent_id    | STR. ID to the parent. Prefix `t1_` if the parent is another comment. Prefix `t3_` if it is a post. |
| score        | INT. Score the comment received.                                                                    |
| subreddit    | STR. Subreddit the comment was made.                                                                |
| author       | STR. The author of the comment.                                                                     |
| slur         | STR. Slur in the comment.                                                                           |
| body         | STR. Body of the comment.                                                                           |
| disagreement | BOOLEAN. `True` if the two annotators did not agree on the label.                                   |
| gold_label   | STR. Final label for the comment.                                                                   |

## Comment Annotation Labels

Each comment was annotated into one of the following five labels. The [annotation guide](https://github.com/networkdynamics/slur-corpus/blob/main/kurrek.2020.slur-guide.pdf) provides additional information on these labels. 

| LABEL | INFO                             | FREQ  |
|-------|----------------------------------|-------|
| DEG   | Derogatory                       | 20531 |
| NDG   | Non Derogatory Non Appropriative | 16729 |
| HOM   | Homonym                          |  1998 |
| APR   | Appropriative                    |   553 |
| CMP   | Noise                            |   189 |

## Citation Information

Please cite our paper in any published work that uses this corpus.

```
@inproceedings{kurrek-etal-2020-towards,
    title = "Towards a Comprehensive Taxonomy and Large-Scale Annotated Corpus for Online Slur Usage",
    author = "Kurrek, Jana and Saleem, Haji Mohammad and Ruths, Derek",
    booktitle = "Proceedings of the Fourth Workshop on Online Abuse and Harms",
    month = "Nov",
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.alw-1.17",
    doi = "10.18653/v1/2020.alw-1.17",
    pages = "138--149",
}
```

- **https://github.com/08Aristodemus24/hate-speech-and-offensive-language.git**



### Automated Hate Speech Detection and the Problem of Offensive Language
Repository for Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM. You read the paper [here](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665).

### NOTE: This repository is no longer actively maintained. Please do not post issues regarding the compatibility of the existing code with new versions of Python or the packages used. I will not accept any pull requests. If you plan to use this data or code in your research, please review the [issues](https://github.com/t-davidson/hate-speech-and-offensive-language/issues), as several Github users have suggested changes or improvements to the codebase.

### 2019 NEWS
We have a new paper on racial bias in this dataset and others, you can read it [here](https://arxiv.org/abs/1905.12516)


***WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.***

You can find our labeled data in the `data` directory. We have included them as a pickle file (Python 2.7) and as a CSV. You will also find a notebook in the `src` directory containing Python 2.7 code to replicate our analyses in the paper and a lexicon in the `lexicons` directory that we generated to try to more accurately classify hate speech. The `classifier` directory contains a script, instructions, and the necessary files to run our classifier on new data, a test case is provided.


***Please cite our paper in any published work that uses any of these resources.***
~~~
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
~~~

***Contact***
We would also appreciate it if you could fill out this short [form](https://docs.google.com/forms/d/e/1FAIpQLSdrPNlfVBlqxun2tivzAtsZaOoPC5YYMocn-xscCgeRakLXHg/viewform?usp=pp_url&entry.1506871634&entry.147453066&entry.1390333885&entry.516829772) if you are interested in using our data so we can keep track of how these data are used and get in contact with researchers working on similar problems.

If you have any questions please contact `thomas dot davidson at rutgers  dot edu`.

- https://paperswithcode.com/dataset/ruddit





## initial data insights:
**Ethos:**
- dataset contains 565 non hate speech comment and 433 hate speech comments
- of the 433 hate speech comments it is divided into classes violence and non-violent hate speech
- distinct labels of 433 comments include gender, race, national origin, disability, religion, SO
- comment is the feature that contains the hate speech feature
- in binary labeled dataset comment is still the feature that contain the comment
- in binary labeled dataset isHate is the target/real y output value that tells whether a comment is hate speech or not
- 1 if hate and 0 if not hate

- only use ff functions

- project uses the hate-speech-and-offensive-language dataset as external dataset
Loading our Binary Data and the binary external dataset D1: Davidson, Thomas, et al. "Automated hate speech detection and the problem of offensive language." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 11. No. 1. 2017.
X, y = Preproccesor.load_data(True)
X_tweets, y_tweets = Preproccesor.load_external_data(True)
class_names = ['noHateSpeech', 'hateSpeech']


- creates the word embeddings as well as trains a neural net with bidirectional lstm archictecture
max_features = 15000
max_len = 150
emb_ma = 1
embed_size = 150

tk = Tokenizer(lower=True, filters='', num_words=max_features, oov_token=True)
tk.fit_on_texts(np.concatenate((X, V)))
tokenized = tk.texts_to_sequences(X)
x_train = pad_sequences(tokenized, maxlen=max_len)
tokenized = tk.texts_to_sequences(V)
x_valid = pad_sequences(tokenized, maxlen=max_len)
embedding_matrix = create_embedding_matrix(emb_ma, tk, max_features)
embedding_matrix.shape
(15001, 300)
We create our model:

file_path = "final.hdf5"
check_point = ModelCheckpoint(
    file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=50)
main_input1 = Input(shape=(max_len,), name='main_input1')
x1 = (Embedding(max_features + 1, 300, input_length=max_len,
                weights=[embedding_matrix], trainable=False))(main_input1)
x1 = SpatialDropout1D(0.4)(x1)
x2 = Bidirectional(LSTM(75, dropout=0.5, return_sequences=True))(x1)
x = Dropout(0.55)(x2)
x = Bidirectional(LSTM(50, dropout=0.5, return_sequences=True))(x)
hidden = concatenate([
    Attention(max_len)(x1),
    Attention(max_len)(x2),
    Attention(max_len)(x)
])
hidden = Dense(32, activation='selu')(hidden)
hidden = Dropout(0.5)(hidden)
hidden = Dense(16, activation='selu')(hidden)
hidden = Dropout(0.5)(hidden)
output_lay1 = Dense(8, activation='sigmoid')(hidden)
model = Model(inputs=[main_input1], outputs=output_lay1)
model.compile(loss="binary_crossentropy", optimizer=Adam(),
              metrics=['binary_accuracy'])


**Breadcrumbs-A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech:**
- text feature contains the hate speech comment
- there is no label for the comment of whether it is hate speech or not, because all rows are designated as hate speech

**slur-corpus:**
- gold label is the label of whether the comment is hate speech or noy
- body is the feature which contains the comment
- comment label consists of about 5 categories

- DEG derogatory 20531
- NDG non derogatory 16729
- HOM homonym 1998
- APR appropriate 553
- CMP noise 189

**hate-speech-and-offensive-language**
- class is the label/target/real y output column
- tweet is the feature which contains offensive comment
- feature columsn hate, offensive, and neither measure to the degree in which a comment is
- hate, offensive, or neither

- 0 for hate I hope you get type 2 diabetes nigger
- 1 for offensive your pussy stinks
- 2 for neither a woman shouldn't complain





## To do:
need to review how to extract only necessary comment itself that includes offensive language in 
need to run preprocess scripts of each repository on each of their respective datasets
**dataset**
- <u>hate-speech-and-offensive-language (done)</u>
- <u>hate class currently 0 can be lumped in with derogatory class of slur dataset, so encode to 2</u>
- <u>offensive class currently 1 can be lumped in with appropriative class of slur dataset, so encode to 0</u>
- <u>neither class currently 2 can be lumped in with non derogatory of slur dataset, so encode to 4</u>

- <u>ethos_data</u>

- <u>slur-corpus</u>
- <u>need to encode labels in order</u>
- <u>probably non derogatory and noise can be just combined so encode noise to 4</u>

- A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech (TBA)

**exploratory data analysis**
Here we investigate the problem of hate speech and ask the following questions which we will eventually answer to help lessen or even outright solve the problem of hate speech
- What defines a slur?
- What would be the statistical treatment used in words?
- what words are most frequently attributed in derogatory comments? Isolate the comments with derogatory label and get each unique word count
- what are the percentages of thsee frequent derogatory comments?
- what words are most frequently attributed in offensive comments? Isolate the comments with derogatory label and get each unique word count
- what are the percentages of thsee frequent offensive comments?
- what words are most frequently attributed in non-derogatory comments? Isolate the comments with derogatory label and get each unique word count
- what are the percentages of thsee frequent non-derogatory comments?
- what are 

- What are the most unusual slurs?
- What is the percentage of these unusual slurs?

- Once this is done highlight the classification problem
- <u>use colormap to visualize first 20 words frequencies in bar chart</u>
- <u>what is the percentages of each first 20 words in each class. use pie chart</u>

**word embedding model**
- still need to tune model since words that are supposed to be similar give low cosine similarity scores
- maybe change parameters of word2vec model like

the effect of min_count and sample on the word corpus. I noticed that these two parameters, and in particular sample, have a great influence over the performance of a model. Displaying both allows for a more accurate and an easier management of their influence.

The threshold for configuring which higher-frequency words are randomly downsampled. Highly influencial. - (0, 1e-5)

negative = int - If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown. If set to 0, no negative sampling is used. - (5, 20)

**sentiment classifier model**
