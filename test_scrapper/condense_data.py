import WE_mag_scrapper
import glob
import json

DATA_DIR = "data"
FILENAME = "winemag-data"

print("Condensing Data...")
condensed_data = []
all_files = glob.glob("{}/*.json".format(DATA_DIR))
for file in all_files:
    with open(file, "rb") as fin:
        condensed_data += json.load(fin)
print(len(condensed_data))
filename = "{}.json".format(FILENAME)
with open(filename, "w") as fout:
    json.dump(condensed_data, fout)
