import pickle

# my unique modules
from nlp_analysis import special_stopwords, normalize, get_wordnet_pos
import user_wine_analysis
import add_features

# import the names of the reviewers
loaded_names = pickle.load(open('reviewer_names.sav', 'rb'))

#last 2 reviewers on list do not have active ML's (not enough data)
loaded_names = loaded_names[0:len(loaded_names)-2]
# remove Sean P. Sullivan due to feature train issues
loaded_names.remove('Sean P. Sullivan')

# generate the specalized stopword list for wine reviewers
stopwords = special_stopwords()
#print(';' in stopwords)

# import the trained ML into a dictionary that we can use on user's wine
#and the feature lists for each ML
model_dic = {the_name:0 for the_name in loaded_names}
feature_dic = {the_name:[] for the_name in loaded_names}
for name in loaded_names:
    # ml's
    filename = 'fits/rf_fit_%s.sav'%name
    rf_fit = pickle.load(open(filename,'rb'))
    model_dic[name] = rf_fit

    # features
    filename = "fits/features_%s.sav"%name
    features = pickle.load(open(filename,'rb'))

    # feature have assiociated # of instances of the words- so remove them
    features = add_features.clean_numbers(features)
    feature_dic[name] = features

# load user sentence from webapp and clean it
some_input = "Splendid ruby color with garnet hues. Chic nose, full of juicy red fruit. On the palate, medium body, crisp acidity with smooth tannins. Strawberry, and raspberry are interlaced with fennel, butter with hints of ripe tomato in a delicious and silky whole."

some_tokens = user_wine_analysis.tokenize_user_input(some_input, stopwords)

#attempt to Lemmatize the users tokens
lemitized_words = normalize(some_tokens)

results_dic = {}
# now have to loop over the names of each reviewer with a ML, as everyhting pecomes person specific now
for name in loaded_names:
    # make cleaned tokens of user sentence into database that can be put through trained ML
    user_df = user_wine_analysis.make_dataframe(feature_dic[name], lemitized_words)

    #get appropriate model
    current_model = model_dic[name]

    #finally run user sentence through ML
    points, probability = user_wine_analysis.run_user_in_ML(user_df, current_model)

    #store the output in a dictionary?/list-of-lists? where {name:[score accuracy]}/[name score accuracy]
    results_dic[name] = int(points[0])

# return the top 5 scores in a list to send to webpage 
top_5_list = user_wine_analysis.top_5_reviewers(results_dic)
