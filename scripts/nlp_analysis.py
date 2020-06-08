import pandas as pd
from pandas import Series,DataFrame

from nltk import sent_tokenize, word_tokenize
import nltk.corpus
from nltk.stem import WordNetLemmatizer

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance

import numpy as np

def stopword_clean(token_pd, stopwords):
# function to clean stopwords from wine descriptions
    for index,row in token_pd.iteritems():
        for y in row:
            if y in stopwords:
                row.remove(y)
    return token_pd

def run_NLP_BOW(first_10):
    # run a nlp "bag of words" on a description

    # get stopwords and add extra stopwords relevant to wine wine reviews
    usr_defined_stop = ['.', ',',"'s", 'is', "n't", '%', 'aromas', 'include', 'wine', 'opens',\
                    'carry', 'note','offers','alongside', 'drink', 'hint', 'dried','delivers','finish','lead', \
                    'firm', 'nose','palate', 'made', 'glass', 'along', 'yellow']
    stop = nltk.corpus.stopwords.words('english')
    stopwords= set(stop).union(usr_defined_stop)

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

    # tokens is a list of list, so making it a single long list
    full_text=[]
    for row in tokens.tolist():
        full_text = full_text + row

    # get the important root words for each wine description word
    lemmzr = WordNetLemmatizer()
    processed_words = []
    for word in full_text:
        processed_words.append(lemmzr.lemmatize(word))

    # get the top 50 most coomon words, and make them a list to be features in data
    freq = nltk.FreqDist(processed_words)
    test_tag_words = freq.most_common(50)

    # return the stopwords, tokenized descriptions, 50 most frequent words
    return stopwords, tokens, test_tag_words


def run_NLP_TFIDF():
    # use a more advanced word processing algorithm
    



    return
