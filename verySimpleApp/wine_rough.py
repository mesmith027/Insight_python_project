
import matplotlib.pyplot as plt #matlab plots
import seaborn as sns
sns.set_style('whitegrid') # style preference on graphs

import pandas as pd
from pandas import Series,DataFrame

from nltk import sent_tokenize, word_tokenize
import nltk.corpus
from nltk.stem import WordNetLemmatizer

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble,tree,model_selection

stopwords = []
lemmzr = WordNetLemmatizer()

def stopword_clean(token_pd):
# function to clean stopwords from wine descriptions
    for index,row in token_pd.iteritems():
        for y in row:
            if y in stopwords:
                row.remove(y)
    return token_pd

def run_ML():
    # load data, and remove an empty column
    wine_130 = pd.read_csv("data/wine_reviews_130k.csv")
    wine_130 = wine_130.loc[:, ~wine_130.columns.str.contains('^Unnamed')]

    # test NLP on a single reviewer
    first_10 = wine_130[wine_130['taster_name'] == 'Kerin O’Keefe']

    # add extra stopwords
    usr_defined_stop = ['.', ',',"'s", 'is', "n't", '%', 'aromas', 'include', 'wine', 'opens',\
                    'carry', 'note','offers','alongside', 'drink', 'hint', 'dried','delivers','finish','lead', \
                    'firm', 'nose','palate', 'made', 'glass', 'along']
    stop = nltk.corpus.stopwords.words('english')
    stopwords= set(stop).union(usr_defined_stop)

    #with added ',','.' and other custom stopwords can go straight to word_tokenize
    tokens = first_10['description'].str.lower().apply(word_tokenize)

    # need to run through multiple times to clean everything properly, misses small words\\
    # on first few passes
    tokens = stopword_clean(tokens)
    tokens = stopword_clean(tokens)
    tokens = stopword_clean(tokens)
    tokens = stopword_clean(tokens)
    tokens = stopword_clean(tokens)
    tokens = stopword_clean(tokens)

    #print("applied stopwords")
    #print(tokens[0:30])

    full_text=[]
    for row in tokens.tolist():
        full_text = full_text + row

    #lemmzr = WordNetLemmatizer()

    processed_words = []
    for word in full_text:
        processed_words.append(lemmzr.lemmatize(word))

    #print('lemmatized words, distinct root words:', len(processed_words))
    freq = nltk.FreqDist(processed_words)

    # make tester words for features -> eventually will be NLP word embedding or bag of words
    test_tag_words = freq.most_common(50)
    #print('number of tag word features:', len(test_tag_words))

    #test_tag_words = ['fruit', 'oak', 'dry', 'sweet', 'light', 'full bodied', 'toasted', 'red', 'rose', 'white',\
    #                  'sparkling', 'herb', 'tannin', 'berry', 'acidity', 'citrus', 'pepper' ]
    # clean up some columns
    first_10 = first_10.drop(['taster_twitter_handle', 'title', 'region_2'], axis=1)

    #test_tag_words is list of (name, # occurances) assign just name as feature
    for x in test_tag_words:
        first_10[x[0]] = 0

    #to use in to add features to dataframe, need to clean out all the numbers
    clean_numbers = []
    for y in test_tag_words:
        clean_numbers.append(y[0])
    test_tag_words = clean_numbers

    # add 1 for tags that appear in each discription
    for index, row in tokens.iteritems():
        for y in row:
            if y in test_tag_words:
                first_10.at[index, y] = 1

#print('added lemmatized features to database, N_features:', len(test_tag_words))
#print(first_10.iloc[0:4,10:60])
# now run some simple linear regression on the features to see if they can \\
# predict the score

#split the data randomly- start with 80% training
    train_feat, test_feat, train_pred, test_pred = train_test_split(first_10.iloc[:,10:60], \
                                                first_10['points'], test_size=0.2, random_state = 42)

#lets run a silly decision tree first to get our feet wet
    dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state= 42)
    fit_dt = dt.fit(train_feat,train_pred)
#print("******************************** Decision Tree *******************************************")
#print(fit_dt.feature_importances_*100)

    dt_r2 = fit_dt.score(train_feat,train_pred)
#print('r^2 value:', dt_r2)

    dt_scores = model_selection.cross_val_score(fit_dt,train_feat,train_pred, scoring='accuracy', cv=10)
#print('average r^2: ',dt_scores.mean())
#print('difference r^2 - aver^2', dt_r2 - dt_scores.mean())

    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=2, random_state=42, n_jobs=-1)
    fit_rf = rf.fit(train_feat,train_pred)

#print("")
#print("")
#print("******************************** Random Forest *******************************************")
#print(fit_rf.feature_importances_*100)

    rf_r2 = fit_rf.score(train_feat,train_pred)
#print("r^2 values for rf:", rf_r2)

    rf_scores = model_selection.cross_val_score(fit_rf,train_feat,train_pred,scoring='accuracy', cv=10)
#print('average r^2: ', rf_scores.mean())
#print('difference r^2 - aver^2', dt_r2 - rf_scores.mean())
    return fit_rf, test_tag_words, stopwords

def stopword_clean_user(some_tokens, stopwords):
    for y in some_tokens:
        if y in stopwords:
            some_tokens.remove(y)

# for week 2 webapp launch use random forest regressor
def get_usr_input(user_input_sentence, test_tag_words, fit_rf, stopwords):
    if type(user_input_sentence) is str:
        user_tokens = word_tokenize(user_input_sentence.lower())
    else:
        user_tokens = word_tokenize(user_input_sentence.str.lower())

    user_tokens = list(user_tokens)
#    print(user_tokens)

    #multiple run through of stopword function to get all stragglers
    tokens = stopword_clean_user(user_tokens, stopwords)
    tokens = stopword_clean_user(user_tokens, stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)
    tokens = stopword_clean_user(user_tokens,stopwords)

    #lemmzr = WordNetLemmatizer()
    for word in user_tokens:
        lemmzr.lemmatize(word)

    #create a pandas array of 0's then fill in 1's for the appropriate user_tokens
#    print(len(test_tag_words))
    user_wine = pd.DataFrame(np.zeros(shape =(1,len(test_tag_words))), columns=test_tag_words)
    for column in user_wine.columns:
        if column in user_tokens:
            user_wine[column] = 1
        else:
            user_wine[column] = 0

    #use only the values from the dataframe and reshape the np.ndarray row into something readable to the fit_rf
    user_array = user_wine.values
#    print(user_array)
    predicted_points = fit_rf.predict(user_array)
    predicted_prob = fit_rf.predict_proba(user_array)
#    print(predicted_points, predicted_prob)

    return 'Kerin O’Keefe', predicted_points, predicted_prob

#fit_rf, tag_words, stopwords = run_ML()
#print("run_ml complete")
#name, points, percentage = get_usr_input('oak red wine with lots of tannins ash and espresso', tag_words, fit_rf, stopwords)
#print(name, points, percentage)
