import pandas as pd
from pandas import Series,DataFrame

#import pdb

import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn import ensemble,tree,model_selection

import pickle

# my unique modules:
import clean_data as cleaning
import nlp_analysis as nlp_words
import add_features
import run_ML

# load the database into pandas
wine_130 = pd.read_csv("../../data/wine_reviews_130k.csv")

#load new 2018 data
wine_test = pd.read_csv("../../data/wine_2018_total.csv")

# clean the database of unwanted columns and reviews with no reviewer name
wine_130 = cleaning.drop_unwanted_columns(wine_130)
wine_130 = cleaning.remove_nan_reviewer(wine_130)

#clean the test set
wine_test = cleaning.remove_nan_reviewer(wine_test)
wine_test = cleaning.drop_unwanted_columns(wine_test)

# create a list of the reviewer's name's to create seperate pandas databases for each
# and be able to run seperte ml algorithm on each person
taster_names = wine_130['taster_name'].unique().tolist()

# save list of reviewer names so we can load it into the web app
file = 'reviewer_names.sav'
pickle.dump(taster_names, open(file,'wb'))

for reviewer_name in taster_names:

    wine_subset = wine_130[wine_130['taster_name'] == reviewer_name]
    wine_test_subset = wine_test[wine_test['taster_name'] == reviewer_name]
    print(reviewer_name)
    print(len(wine_test_subset))

    # with person specific data, run NLP_analysis to get stopwords, tokenized descriptions, 50 most frequent words
    custom_stopwords, wine_token_descriptions, top_50_words = nlp_words.run_NLP_BOW(wine_subset)
    #print(type(wine_token_descriptions))

    # retrieve the tokens from the test data set
    test_tokens = nlp_words.run_test_tokenize(wine_test_subset)

    # make pandas database from top 50 words on training and test set
    wine_subset = add_features.add_token_features(wine_subset, wine_token_descriptions, top_50_words)
    wine_test_subset = add_features.add_token_features(wine_test_subset, test_tokens, top_50_words)

    # now that we have the relevant tokens for each person, run mL algorithm
    # split the dataframe into features and prediction
    train_feat = wine_subset.iloc[:,8:]
    train_pred = wine_subset['points']
    test_feat = wine_test_subset.iloc[:,9:]
    test_pred = wine_test_subset['points']

    #run ML of choice
    #fit_ml = run_ML.random_forest_Clas(train_feat,train_pred,test_feat,test_pred)
    fit_ml = run_ML.random_forest_Reg(train_feat,train_pred,test_feat,test_pred)

    # save the trained ML using pickle
    filename = 'fits/rf_fit_%s.sav'%reviewer_name
    pickle.dump(fit_ml, open(filename,'wb'))

    #save the relevant features for each reviewer
    filename = 'fits/features_%s.sav'%reviewer_name
    pickle.dump(top_50_words, open(filename,'wb'))
