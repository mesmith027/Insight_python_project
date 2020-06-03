
import matplotlib.pyplot as plt #matlab plots
import seaborn as sns
sns.set_style('whitegrid') # style preference on graphs

import pandas as pd
from pandas import Series,DataFrame

from nltk import sent_tokenize, word_tokenize
import nltk.corpus

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble,tree,model_selection

def stopword_clean(token_pd):
# script to clean stopwords from wine descriptions
    for index,row in token_pd.iteritems():
        for y in row:
            if y in stopwords:
                row.remove(y)
    return token_pd

# load data, and remove an empty column
wine_130 = pd.read_csv("../data/wine_reviews_130k.csv")
wine_130 = wine_130.loc[:, ~wine_130.columns.str.contains('^Unnamed')]

# test NLP on a single reviewer
first_10 = wine_130[wine_130['taster_name'] == 'Kerin O’Keefe']

# add extra stopwords
usr_defined_stop = ['.', ',',"'s", 'is', "n't"]
stop = nltk.corpus.stopwords.words('english')
stopwords= set(stop).union(usr_defined_stop)

#with added ',','.' and other custom stopwords can go straight to word_tokenize
tokens = first_10['description'].str.lower().apply(word_tokenize)

# need to run through 3 times to clean everything properly, misses small words\\
# on first 2 passes
tokens = stopword_clean(tokens)
tokens = stopword_clean(tokens)
tokens = stopword_clean(tokens)

full_text=[]
for row in tokens.tolist():
    full_text = full_text + row

freq = nltk.FreqDist(full_text)

# make tester words for features -> eventually will be NLP word embedding or bag of words
test_tag_words = freq.most_common(20)

#test_tag_words = ['fruit', 'oak', 'dry', 'sweet', 'light', 'full bodied', 'toasted', 'red', 'rose', 'white',\
#                  'sparkling', 'herb', 'tannin', 'berry', 'acidity', 'citrus', 'pepper' ]
# clean up some columns
first_10 = first_10.drop(['taster_twitter_handle', 'title', 'region_2'], axis=1)

#add my test_tag_words as features in data
for x in test_tag_words:
    first_10[x] = 0

# add 1 for tags that appera in discription
for index, row in tokens.iteritems():
    for y in row:
        if y in test_tag_words:
            first_10.at[index, y] = 1

#print(first_10.iloc[:, 10:26])
# now run some simple linear regression on the features to see if they can \\
# predict the score

#split the data randomly- start with 80% training
train_feat, test_feat, train_pred, test_pred = train_test_split(first_10.iloc[:,10:26], first_10['points'], test_size=0.2, random_state = 42)

#lets run a silly decision tree first to get our feet wet
dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state= 42)
fit_dt = dt.fit(train_feat,train_pred)
print("******************************** Decision Tree *******************************************")
print(fit_dt.feature_importances_*100)

dt_r2 = fit_dt.score(train_feat,train_pred)
print('r^2 value:', dt_r2)

dt_scores = model_selection.cross_val_score(fit_dt,train_feat,train_pred, scoring='accuracy', cv=10)
print('average r^2: ',dt_scores.mean())
print('difference r^2 - aver^2', dt_r2 - dt_scores.mean())

rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, random_state=42, n_jobs=-1)
fit_rf = rf.fit(train_feat,train_pred)
print("******************************** Random Forest *******************************************")
print(fit_rf.feature_importances_*100)

rf_r2 = fit_rf.score(train_feat,train_pred)
print("r^2 values for rf:", rf_r2)

rf_scores = model_selection.cross_val_score(fit_rf,train_feat,train_pred,scoring='accuracy', cv=10)
print('average r^2: ', rf_scores.mean())
print('difference r^2 - aver^2', dt_r2 - rf_scores.mean())
