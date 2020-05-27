# Insight_python_project
Python repo for insight data science internship project and general work


---
## Directories
- **./data**
  - holds the possible datasets that I may use for the final project
  - **wine_reviews_150K.csv:** wine reviews from https://www.kaggle.com/
    - **wine_reviews_150K.csv:** smaller wine data set that is known to have duplicates
  - **ks_projects.csv:** kickstarter data of successful/cancelled/failed projects from https://www.kaggle.com/
  - **ks_most_backed.csv:** kiskstarter data for the most successful projects
  - **ks_live.csv:** kickstarter data on 'live' projects as of 2016-10-29 5pm PDT

---
## What you need to run

- This is run through the terminal
    - open a terminal and navigate to this folder
    -  to run `python sentdex_youtube_tutorial.py`


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
