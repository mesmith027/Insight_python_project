from flask import Flask, render_template, request
import pickle

# my unique modules
from scripts.nlp_analysis import special_stopwords, normalize, get_wordnet_pos
import scripts.user_wine_analysis as user_wine_analysis
import scripts.add_features as add_features

import os.path

# import the names of the reviewers
loaded_names = pickle.load(open('scripts/reviewer_names.sav', 'rb'))
print(loaded_names)
#last 2 reviewers on list do not have active ML's (not enough data)
loaded_names = loaded_names[0:len(loaded_names)-2]
# remove Sean P. Sullivan due to feature train issues
loaded_names.remove('Sean P. Sullivan')

# generate the specalized stopword list for wine reviewers
stopwords = special_stopwords()

# import the trained ML into a dictionary that we can use on user's wine
#and the feature lists for each ML
model_dic = {the_name:0 for the_name in loaded_names}
feature_dic = {the_name:0 for the_name in loaded_names}
for name in loaded_names:
    # ml's
    filename = 'scripts/fits/rf_fit_%s.sav'%name
    rf_fit = pickle.load(open(filename,'rb'))
    model_dic[name] = rf_fit

    # features
    filename = "scripts/fits/features_%s.sav"%name
    features = pickle.load(open(filename,'rb'))

    # feature have assiociated # of instances of the words- so remove them
    features = add_features.clean_numbers(features)
    feature_dic[name] = features

# Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input

def home_page():
        return render_template('index.html')

@app.route('/output')

def recommendation_output():
        # Pull input
        some_input =request.args.get('user_input')

        # Case if empty
        if some_input =="":
                return render_template("index.html",my_input = some_input, my_form_result="Empty")
        else:
                #render_template("index.html",my_input = some_input, my_form_result="Processing")
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

                person_1 = top_5_list[0][0]
                score_1 = top_5_list[0][1]
                person_2 = top_5_list[1][0]
                score_2 = top_5_list[1][1]
                person_3 = top_5_list[2][0]
                score_3 = top_5_list[2][1]
                person_4 = top_5_list[3][0]
                score_4 = top_5_list[3][1]
                person_5 = top_5_list[4][0]
                score_5 = top_5_list[4][1]
                print(person_1)
                #some_image="./img/pour_wine_3.jpg"

                # get profile image if exists
                if os.path.exists("./static/wine_profile/%s.png"%person_1):
                    profile_file_name = "./wine_profile/%s.png"%person_1
                else:
                    profile_file_name = "./wine_profile/nothing_found.png"
                print(profile_file_name)
                some_image=profile_file_name

        # return reviewers and profile picture to index 
        return render_template("index.html", my_input=some_input,output_1=person_1,number_1=score_1,\
                                output_2=person_2,number_2=score_2,\
                                output_3=person_3,number_3=score_3,\
                                output_4=person_4,number_4=score_4,\
                                output_5=person_5,number_5=score_5,\
                                my_img_name=some_image,my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
