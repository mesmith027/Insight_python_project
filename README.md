# Insight_python_project
Python repo for insight data science internship project and general work.
the main web app is the the verySimpleApp folder. In that folder there is a directory that houses all the scripts to analyze the data and generate trained ML algorithms that can be loaded into the app and used to generate the predicted score of a wine based on its description.

---
## Directories
list of directories, what they are and the files in them
- **notebooks:** jupyter notebooks of the narrative of the project and the direction it has taken
  - **QL_wine.ipynb:** Quick Look at the wine reviews dataset, looking if there is enough data and any signal to develop a project on
  - **Isaac_ws_test.ipynb:** Isaac Chung's web-scraping test of auto trader that can be used as an example
  - **JSON_to_CSV_test.ipynb:** Test Jupyter notebook that looks at importing a json file and turning it into a csv file
  - **regressor_accuracy_graphs_working.ipynb:** file that graphs the test accuracy of the trained ML models (currently random forest regressor) and graph the predicted and actual values for the points score of the wine
  - **TDIDF_example.ipynb:** example file of how to use TFIDF to implement in future versions of the scripts and ML portion of the code

- **data**
  - holds the possible datasets that I may use for the final project
  - **wine_reviews_150K.csv:** wine reviews from https://www.kaggle.com/
    - **wine_reviews_130K.csv:** smaller wine data set that is known to have duplicates
  - **ks_projects.csv:** kickstarter data of successful/cancelled/failed projects from https://www.kaggle.com/
  - **ks_most_backed.csv:** kickstarter data for the most successful projects
  - **ks_live.csv:** kickstarter data on 'live' projects as of 2016-10-29 5pm PDT

- **verySimpleApp:** simple webApp that loads the files from the scripts folder and can run locally or as an instance on AWS
  - **scripts:** scripts that clean the data, run NLP and the chosen ML, and save the trained ML algorithms to ./fits/ folder
    - **ML_training_sript.py:** trains selected ML algorithm (random forest regressor)
    - **webapp_test_script.py:** used to test the sentence user input of the website on the trained ML algorithms
    - **fits/** directory that houses all the trained ML algorithms to be loaded into the web app and the predicted_test outcomes so they can be visualized
    - **depreciated:** holds old code no longer used
  - **static:** holds all the formatting from the download template of the website
  - **templates:** holds the index.html file that is the main file to run the web app

- **test_scraper:** holds the downloaded test scraper for scraping online wine magazine Wine Enthusiast
  - **condensed_data:** holds the data files, json and csv
  - **WE_mag_scrapper.py:** the actual scraper script by Zackthoutt (git username)
  - **json_to_csv.py:** script to convert json files to csv files
  - **condense_data.py:** script to convert data stored in auto generated ./data/ folder into one json file, if the web scrapper gets denied and has to be stopped before it's finished scraping. ./data/ folder can be deleted at the end of this manually

---
## What you need to run
- **NLTK:** natural language processing kit for python
  - installed with: conda install -c anaconda nltk
  - one time download of:
    - nltk.download('stopwords')
    - nltk.download('punkt')
    - nltk.download('wordnet')
    - nltk.download('averaged_perceptron_tagger')

- **Flask:**
  - install:  pip install flask
  - conda install -c anaconda flask

- **Pickle:**
  - used for storing trained ML algorithms
  - install: conda install -c conda-forge pickle5

- **SK-learn:** to train ML algorithms
  - conda install scikit-learn

- **Pandas:**
  - database package to manage the data
  - conda install -c anaconda pandas

- **Numpy:**
  - conda install -c anaconda numpy

- **JSON:**
  - conda install -c jmcmurray json

- **CSV:**
  - conda install -c anaconda csvkit


---
## Git Repository

This is a repo that is located on BitBucket, which runs git.
Important instructions:

- git status: shows the status of the files changed in the local dir.
- git pull: pulls from the repo the latest changes if working with a
    group/ multiple computers- this is important to avoid merge errors
- git add <filename> or git add .
    - <filename> adds that specific file to be committed
    - . adds all the changed files
- git commit -m 'message for commit'
    - without the -m then git will use the default editor to force a
    commit message, which is usually vi/vim (eww)
- git push
    - pushed all the changes from the local computer to the repo
