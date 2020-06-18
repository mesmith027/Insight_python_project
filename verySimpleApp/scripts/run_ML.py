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

def random_forest_Clas(train_feat, train_pred, test_feat, test_pred):
    # create a trained enesmble random forest clasifier
    #train_values = train_feat.values
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=2, random_state=42, n_jobs=-1)
    fit_rf = rf.fit(train_feat,train_pred)

    rf_r2 = fit_rf.score(train_feat,train_pred)
    rf_scores = model_selection.cross_val_score(fit_rf,train_feat,train_pred,scoring='accuracy', cv=5)
    print('r^2 value:', rf_r2)
    print('average r^2: ', rf_scores.mean())
    print('difference r^2 - aver^2', rf_r2 - rf_scores.mean())

    if len(test_feat) == 0:
        print("WARNING: no current test set in scrapped data")
    # run on test data
    else:
        if 1 <= len(test_feat) <= 10:
            print("Warning: low numbers in validation set")
        test_values = test_feat.values
        predicted_score = fit_rf.predict(test_feat)

        #validate againt actual test_pred
        errors = abs(predicted_score-test_pred)
        print('errors:', errors, sep='\n')

        # calculate Mean Absolute Percentage Error
        mape = 100*(errors/test_pred)

        # accuracy
        accuracy = 100-np.mean(mape)
        print('accuracy:', accuracy)

    return fit_rf

def random_forest_Reg(train_feat, train_pred, test_feat, test_pred):
    # create trained ensemble random forest regressor
    rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=2, random_state=42, n_jobs=-1)
    fit_rf = rf.fit(train_feat,train_pred)

    rf_r2 = fit_rf.score(train_feat,train_pred)
    rf_scores = model_selection.cross_val_score(fit_rf,train_feat,train_pred,scoring='r2', cv=5)
    print('r^2 value:', rf_r2)
    print('RMS error: ', rf_scores.mean())
    print('difference r^2 - aver^2', rf_r2 - rf_scores.mean())

    if len(test_feat) == 0:
        print("WARNING: no current test set in scrapped data")
        predicted_score =[]
        accuracy = 0
    # run on test data
    else:
        if 1 <= len(test_feat) <= 10:
            print("CAUTION: low numbers in validation set")
        test_values = test_feat.values
        predicted_score = fit_rf.predict(test_feat)

        #validate againt actual test_pred
        errors = abs(predicted_score-test_pred)

        # calculate Mean Absolute Percentage Error
        mape = 100*(errors/test_pred)

        # accuracy
        accuracy = 100-np.mean(mape)
        print('accuracy:', accuracy)

    return fit_rf, predicted_score, accuracy
