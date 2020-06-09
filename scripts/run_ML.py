import pandas as pd
from pandas import Series,DataFrame

from sklearn.model_selection import train_test_split
from sklearn import ensemble,tree,model_selection, feature_extraction

import numpy as np

def decision_tree(train_feat, train_pred, test_feat, test_pred):
    # lets run a silly decision tree first to get our feet wet
    dt = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state= 42)
    fit_dt = dt.fit(train_feat,train_pred)

    dt_r2 = fit_dt.score(train_feat,train_pred)
    dt_scores = model_selection.cross_val_score(fit_dt,train_feat,train_pred, scoring='accuracy', cv=10)
    print('r^2 value:', dt_r2)
    print('average r^2: ', dt_scores.mean())
    print('difference r^2 - aver^2', dt_r2 - dt_scores.mean())

    # run on test data
    return fit_dt

def random_forest(train_feat, train_pred, test_feat, test_pred):
    # create a trained enesmble random forest clasifier
    #train_values = train_feat.values
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=2, random_state=42, n_jobs=-1)
    fit_rf = rf.fit(train_feat,train_pred)

    rf_r2 = fit_rf.score(train_feat,train_pred)
    rf_scores = model_selection.cross_val_score(fit_rf,train_feat,train_pred,scoring='accuracy', cv=10)
    print('r^2 value:', rf_r2)
    print('average r^2: ', rf_scores.mean())
    print('difference r^2 - aver^2', rf_r2 - rf_scores.mean())

    # run on test data
    test_values = test_feat.values
    #print(test_feat)
    predicted_score = fit_rf.predict(test_feat)
    #print(predicted_score)

    #validate againt actual test_pred
    errors = abs(predicted_score-test_pred)
    print('errors:', errors, sep='\n')

    # calculate Mean Absolute Percentage Error
    mape = 100*(errors/test_pred)

    # accuracy
    accuracy = 100-np.mean(mape)
    print('accuracy:', accuracy)

    return fit_rf
