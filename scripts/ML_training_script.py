import pandas as pd
from pandas import Series,DataFrame

#from nltk import sent_tokenize, word_tokenize
#import nltk.corpus
#from nltk.stem import WordNetLemmatizer

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
wine_130 = pd.read_csv("../data/wine_reviews_130k.csv")

# clean the database of unwanted columns and reviews with no reviewer name
wine_130 = cleaning.drop_unwanted_columns(wine_130)
wine_130 = cleaning.remove_nan_reviewer(wine_130)

# create a list of the reviewer's name's to create seperate pandas databases for each
# and be able to run seperte ml algorithm on each person
taster_names = wine_130['taster_name'].unique().tolist()

# save list of reviewer names so we can load it into the web app
file = 'reviewer_names.sav'
pickle.dump(taster_names, open(file,'wb'))

print(len(taster_names))

#reviewer_name = taster_names[0]
for reviewer_name in taster_names:
    wine_subset = wine_130[wine_130['taster_name'] == reviewer_name]
    print(reviewer_name)
    # with preson specific data, run NLP_analysis to get stopwords, tokenized descriptions, 50 most frequent words
    custom_stopwords, wine_token_descriptions, top_50_words = nlp_words.run_NLP_BOW(wine_subset)
    #print(top_50_words)

    # make pandas database from top 50 words
    wine_subset = add_features.add_token_features(wine_subset, wine_token_descriptions, top_50_words)
    #print(wine_subset.info())
    #print(wine_subset.head())

    # now that we have the relevant tokens for each person, run mL algorithm
    # split the dataframe
    train_feat, test_feat, train_pred, test_pred = train_test_split(wine_subset.iloc[:,8:len(wine_subset.columns)-1], wine_subset['points'], test_size=0.2, random_state = 42)

    #run ML of choice
    fit_ml = run_ML.random_forest(train_feat,train_pred,test_feat,test_pred)

    # save the trained ML using pickle
    filename = 'fits/rf_fit_%s.sav'%reviewer_name
    pickle.dump(fit_ml, open(filename,'wb'))
