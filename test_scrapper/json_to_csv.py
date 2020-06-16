import json
import csv

# Opening JSON file and loading the data
# into the variable data
def add_row(data, count):
    # Counter variable used for writing
    # headers to the CSV file
    
    for wine_row in data:
        if count == 0:

            # Writing headers of CSV file
            header = wine_row.keys()
            csv_writer.writerow(header)
            count += 1

        # Writing data of CSV file
        csv_writer.writerow(wine_row.values())
    return

with open('winemag-data.json') as json_file:
    data = json.load(json_file)
with open('winemag-data_1_100.json') as json_file:
    more_data = json.load(json_file)

print(data[0])
print(type(data))
print(more_data[0])
print(type(more_data))
#wine_data = data['emp_details']

# now we will open a file for writing
target_file = open('wine_2018_total.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(target_file)

add_row(data, 0)
add_row(more_data, 1)

target_file.close()

import pandas as pd
wine_new = pd.read_csv("wine_2018_total.csv")
print(wine_new.head())
print(wine_new.info())

print(wine_new['taster_name'].value_counts())
