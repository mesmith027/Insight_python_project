import pandas as pd
from pandas import Series,DataFrame

import nltk.corpus
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#nltk.download('averaged_perceptron_tagger')
#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance

import numpy as np

def stopword_clean(token_pd, stopwords):
# function to clean stopwords from wine descriptions
    filtered_tokens = []
    for row in token_pd:
        new_row = row
        for y in row:
            if y in stopwords:
                new_row.remove(y)
        filtered_tokens.append(new_row)
    return filtered_tokens

def special_stopwords():
    # get stopwords and add extra stopwords relevant to wine wine reviews
    usr_defined_stop = ['.', ',',';',"'s", 'is', "n't", '%', 'aromas', 'include', 'wine', 'opens',\
                    'carry', 'note','offers','alongside', 'drink', 'hint', 'dried','delivers','finish','lead', \
                    'firm', 'nose','palate', 'made', 'glass', 'along', 'yellow', "'ll"]
    stop = nltk.corpus.stopwords.words('english')
    stopwords= set(stop).union(set(usr_defined_stop))
    return stopwords

def remove_numbers():

    return

#map NLTK’s POS tags
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#Lemmatize Normalization
def normalize(tokens):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(token,pos=get_wordnet_pos(token)) for token in tokens]


def run_NLP_BOW(first_10):
    # run a nlp "bag of words" on a description

    # get stopwords and add extra stopwords relevant to wine wine reviews
    stopwords = special_stopwords()

    #with added ',','.' and other custom stopwords can go straight to word_tokenize
    tokens = first_10['description'].str.lower().apply(word_tokenize)

    # need to run through multiple times to clean everything properly, misses small words\\
    # on first few passes
    tokens = stopword_clean(tokens, stopwords)
    tokens = stopword_clean(tokens, stopwords)
    tokens = stopword_clean(tokens, stopwords)
    tokens = stopword_clean(tokens, stopwords)
    tokens = stopword_clean(tokens, stopwords)
    tokens = stopword_clean(tokens, stopwords)

    # tokens is a list of lists, so making it a single long list
    full_text=[]
    for row in tokens:
        full_text = full_text + row

    # get the important root words for each wine description word
    #lemmzr = nltk.WordNetLemmatizer()
    #processed_words = []
    #for word in full_text:
    #    print(lemmzr.lemmatize(word))
    #    processed_words.append(lemmzr.lemmatize(word))

    processed_words = normalize(full_text)
    #print(processed_words)
    # get the top 50 most common words, and make them a list to be features in data
    freq = nltk.FreqDist(processed_words)
    test_tag_words = freq.most_common(50)

    # return the stopwords, tokenized descriptions, 50 most frequent words
    return stopwords, tokens, test_tag_words


def run_sklearn_TFIDF():
    # use a more advanced word processing algorithm




    return
