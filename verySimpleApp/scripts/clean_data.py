import pandas as pd
from pandas import Series,DataFrame

# get rid of nan's from reviewers names
def remove_nan_reviewer(pds_data):
    # get rid of nan's from reviewers names
    pds_data = pds_data[pds_data['taster_name'].notna()]
    return pds_data

# dropping columns from wine data that are useless
def drop_unwanted_columns(pds_data):
    # drop unwanted unnamed columns
    pds_data = pds_data.loc[:, ~pds_data.columns.str.contains('^Unnamed')]
    pds_data = pds_data.drop(['taster_twitter_handle', 'title', 'region_2', 'designation', 'region_1'], axis=1)
    return pds_data
