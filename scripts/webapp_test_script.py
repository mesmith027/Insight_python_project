import pickle

# my unique modules
from nlp_analysis import special_stopwords
import user_wine_analysis

# import the names of the reviewers
loaded_names = pickle.load(open('reviewer_names.sav', 'rb'))
print(loaded_names)
print(type(loaded_names))
loaded_names = loaded_names[0:len(loaded_names)-2]
# generate the specalized stopword list for wine reviewers
stopwords = special_stopwords()
print(';' in stopwords)

# import the trained ML into a dictionary that we can use on user's wine
model_dic = {the_name:0 for the_name in loaded_names}
for name in loaded_names:
    filename = 'fits/rf_fit_%s.sav'%name
    rf_fit = pickle.load(open(filename,'rb'))
    model_dic[name] = rf_fit

print(model_dic[loaded_names[1]])

# load user sentence from webapp and run through each ML
#store the output in a dictionary?/list-of-lists? where {name:[score accuracy]}/[name score accuracy]
