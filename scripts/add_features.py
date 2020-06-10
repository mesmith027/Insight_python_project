import pandas as pd
from pandas import Series,DataFrame
#import pdb

def add_token_features(pds_data, description_words, token_words):
    #test_tag_words is list of (name, # occurances) assign just name as feature
    for x in token_words:
        pds_data[x[0]] = 0

    print(pds_data.info())
    #to use in to add features to dataframe, need to clean out all the numbers
    clean_numbers = []
    for y in token_words:
        clean_numbers.append(y[0])
    token_words = clean_numbers

    # add 1 for tags that appear in each discription
    #pdb.set_trace()
    pds_data= pds_data.reset_index(drop=True)
    print(pds_data.head())
    index = -1
    print(len(description_words))
    print(len(pds_data))
    for row in description_words:
        index += 1
        #print(row)
        for word in row:
            #print(index, word)
            if word in token_words:
                pds_data.at[index, word] = 1
        #print(pds_data.iloc[index,8:15])

    return pds_data
