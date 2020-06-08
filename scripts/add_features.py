import pandas as pd
from pandas import Series,DataFrame

def add_token_features(pds_data, description_words, token_words):
    #test_tag_words is list of (name, # occurances) assign just name as feature
    for x in token_words:
        pds_data[x[0]] = 0

    #to use in to add features to dataframe, need to clean out all the numbers
    clean_numbers = []
    for y in token_words:
        clean_numbers.append(y[0])
    token_words = clean_numbers

    # add 1 for tags that appear in each discription
    for index, row in description_words.iteritems():
        for y in row:
            if y in token_words:
                pds_data.at[index, y] = 1

    return pds_data
