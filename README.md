# Insight_python_project
Python repo for insight data science internship project and general work.
The main web app is in the verySimpleApp folder. In that folder there is a directory that houses all the scripts to analyze the data and generate trained ML algorithms that can be loaded into the app and used to generate the predicted score of a wine based on its description.

## The Problem
The wine industry spends a lot of money sending wine to reviewers/tasters and competitions each year. A good review or competition win can translate to an increase in revenue for that wine by $60,000 USD and increase the demand for longer than advertising. The problem is that wine tasting is known to be a junk science with reviews from different reviewers being wildly different for the same wine, as well as the same reviewer rating the same wine differently. So the problem then becomes, why do these reviews vary so widely? And from a business perspective, how can we increase our revenue, or decrease or losses from competitions/bad reviews?

## The Question:
My question then became: maybe the reason that reviews vary so much from reviewer to reviewer is that you can never truly be impartial, and the flavours and types of wine that a particular reviewer likes to drink in their spare time is affecting their wine rating score? If this is the case, I then reasoned it might be possible to predict what rating they will give a wine based on their previous wine rating history.

If I can predict the score a wine will receive based on which reviewer is tasting it, then effectively, they can be sent wines with descriptions that align with their known preferences and have a high probability to get a good rating or receive a winning score at a competition. At competitions, the reviewers/wine tasters that will be the judges are published well in advance, so tailoring the wine submission from a big wine brand is not difficult.   

### How do wine ratings work?
Wine is rated on a scale of 0-100, 100 being the best and 0 being terrible. However, common knowledge in the wine rating industry is that anything that is rated below 80 is deemed (depending on the source) either "undrinkable" or "not-recommended". Therefore, the most commonly seen ratings are on the 20 point scale between 80 (just passed) to 100 (perfect score). Which is sometimes why some wine is rated on a scale of 0-20, this actually represents the number of points added to 80. For example on the "20 point scale" a rating of 0 is actually: 80+0=80, and a rating of 10 is: 80+10=90.

Too be considered a good review or a competition win in most cases/competitions the wine must receive a score of 90 (10) or above. See the Image for the most common scales used in the wine industry.

![Popular wine scales compared](readme_imgs/wine_scale.jpg?raw=true)

## Goal
The main goal of this project is to test my hypothesis; that there are detectable trends in how a wine taster rates wines based on the described flavours and aromas from their wine rating history. The description of a wine categorizes the inherent flavours and aromas of a wine that are detectable when you taste it. Look on the back of any wine bottle and you will see such descriptor words such as: apple, pear, chocolate, tannins, dry, sweet etc.. These words all describe the taste and smell of the wine, and will be used to try and predict a wine score.

For this to work, each reviewer will have their own trained machine learning algorithm based on their unique wine rating history. Then each individual algorithm will be loaded into a webbapp where a description of a wine can be put into a field and a table with the reviewers names and top 5 scores will be generated based on that wines' description.

## Data
The data was scraped from a well known wine magazine called: Wine Enthusiast. I used a 3 year old scrapped data set from Kraggle as the training set and then scrapped new data from the Wine Enthusiast to be used as a testing set.

The training data looks like the picture below:
![Training Data Set](readme_imgs/data.png?raw=true)  

As you can see it consists of rows of wine reviews from various reviewers as well as the description of that wine's taste and aroma. For now I will ignore the other categories and focus just on the wine's description.

### Exploratory Analysis
I wanted to check a few things before I really got started.
1. The number of reviews for each reviewer: if the number is too small then I wont get a good fit and will discard them from my training set.
2. Do all the reviews scrapped have an associated reviewer name: if there are nan's in the name field then these will be discarded.
3. What are the distributions of the wine ratings: if the data is severely unbalanced (many more reviews in 80's and not many in the 90's) then I may have to use some oversampling techniques to help with this
4. I want to look at the frequency of the top 50 words: I want to make sure that after I have run a standard NLP pipeline on the descriptions that we have words that make sense to train on for the wine industry

#### 1 and 2:
Below is a bar graph of the number of reviews for each reviewer name. In fact, there are quite a few reviews that do not have an associated reviewer name with them and so these will have to be discarded. For the last two names on the graph, Fiona Adams and Christina Pickard, there is not likely enough reviews to create a good ML fit. So for now I will train with the other 17 reviewers that I have better statistics on.
![Number of Reviews](readme_imgs/reviewers.png?raw=true)

#### 3:
Below is a histogram for the distribution of the scores for all the wine reviews. Luckily, it looks very normally distributed, so I will not need to do any oversampling techniques for unbalanced datasets, since "wins" are 90 and above and "losses" are 89 and below.
![Distribution of Scores](readme_imgs/number_of_scores.png?raw=true)

#### 4:
This is the frequency (number of counts) after running a standard Bag-of-Words on the wine descriptions. The words all seem very relevant to the wine tasting industry, although some many need to be removed before training as they are not likely to be useful (such as glass and whiff).
![Frequency of Words](readme_imgs/descriptor_counts.png?raw=true)

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
    - **wine_reviews_130K.csv:** smaller wine data set that actually has the wine reviewer's names (taster_name)
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
