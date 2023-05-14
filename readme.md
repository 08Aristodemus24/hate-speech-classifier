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
dataset contains 565 non hate speech comment and 433 hate speech comments
of the 433 hate speech comments it is divided into classes violence and non-violent hate speech
distinct labels of 433 comments include gender, race, national origin, disability, religion, SO
comment is the feature that contains the hate speech feature
in binary labeled dataset comment is still the feature that contain the comment
in binary labeled dataset isHate is the target/real y output value that tells whether a comment is hate speech or not

**Breadcrumbs-A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech:**
text feature contains the hate speech comment
there is no label for the comment of whether it is hate speech or not, because all rows are designated as hate speech

**slur-corpus:**
gold label is the label of whether the comment is hate speech or noy
body is the feature which contains the comment
comment label consists of about 5 categories

derogatory 20531
non derogatory 16729
homonym 1998
appropriate 553
noise 189

**hate-speech-and-offensive-language**
class is the label/target/real y output column
tweet is the feature which contains offensive comment
feature columsn hate, offensive, and neither measure to the degree in which a comment is
hate, offensive, or neither





## To do:
need to review how to extract only necessary comment itself that includes offensive language in 
need to run preprocess scripts of each repository on each of their respective datasets