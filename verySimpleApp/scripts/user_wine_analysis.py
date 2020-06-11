import pandas as pd
from pandas import Series,DataFrame

import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

import operator


def stopword_clean_user(some_tokens, stopwords):
    # remove stopwords from the user's sentence
    for y in some_tokens:
        if y in stopwords:
            some_tokens.remove(y)
    return some_tokens

# for week 3 webapp launch use random forest regressor
def all_in_one(user_input_sentence, test_tag_words, fit_rf, stopwords):
    # clean and tokenize user_input_sentence from webapp
    if type(user_input_sentence) is str:
        user_tokens = word_tokenize(user_input_sentence.lower())
    else:
        user_tokens = word_tokenize(user_input_sentence.str.lower())

    user_tokens = list(user_tokens)

    #multiple run through of stopword function to get all stragglers
    tokens = stopword_clean_user(user_tokens, stopwords)
    print(tokens)
    tokens = stopword_clean_user(user_tokens, stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)

    # lemmatize each relevant word from the user
    for word in user_tokens:
        lemmzr.lemmatize(word)

    #create a pandas array of 0's then fill in 1's for the appropriate user_tokens
    user_wine = pd.DataFrame(np.zeros(shape =(1,len(test_tag_words))), columns=test_tag_words)
    for column in user_wine.columns:
        if column in user_tokens:
            user_wine[column] = 1
        else:
            user_wine[column] = 0

    # use only the values from the dataframe and reshape the np.ndarray row into something readable to the fit_rf
    user_array = user_wine.values
    predicted_points = fit_rf.predict(user_array)
    predicted_prob = fit_rf.predict_proba(user_array)

    return 'Kerin O’Keefe', predicted_points, predicted_prob

def tokenize_user_input(user_input_sentence, stopwords):
    # tokenize the user sentence
    if type(user_input_sentence) is str:
        user_tokens = word_tokenize(user_input_sentence.lower())
    else:
        user_tokens = word_tokenize(user_input_sentence.str.lower())

    user_tokens = list(user_tokens)

    # get rid of the stopwords
    tokens = stopword_clean_user(user_tokens, stopwords)
    tokens = stopword_clean_user(user_tokens, stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)

    return tokens

def make_dataframe(feature_words, user_words):
    #create a pandas array of 0's then fill in 1's for the appropriate user_tokens
    user_wine = pd.DataFrame(np.zeros(shape =(1,len(feature_words))), columns=feature_words)
    for column in user_wine.columns:
        if column in user_words:
            user_wine[column] = 1
        else:
            user_wine[column] = 0
    return user_wine

def run_user_in_ML(user_wine, model):
    # use only the values from the dataframe and reshape the np.ndarray row into something readable to the fit_rf
    user_array = user_wine.values    
    predicted_points = model.predict(user_array)
    predicted_prob = model.predict_proba(user_array)
    return predicted_points, predicted_prob


def top_5_reviewers(reviewer_dic):
    # take generated dictionary and find the top 5 scores and the names of those reviewers an rank them in a list
    # sort the dictionary by decending values
    sorted_d = dict(sorted(reviewer_dic.items(), key=operator.itemgetter(1),reverse=True))

    #get the top 5
    first5pairs = {k: sorted_d[k] for k in list(sorted_d)[:5]}

    #put them in list
    sorted_list = []
    for key,value in first5pairs.items():
        sorted_list.append([key, value])

    return sorted_list
