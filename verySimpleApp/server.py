from flask import Flask, render_template, request
import wine_rough as functions
import pickle

#load in all trained ML algorithms and a list of names of reviewers to use below
#fit_rf, tag_words, stopwords = functions.run_ML()

#load list of top 5 features for each reviewer to make tick boxes on interface

# Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input

def home_page():
        return render_template('index.html')

@app.route('/output')

def recommendation_output():
        # Pull input
        #some_input =request.args.get('user_input')
        some_input =request.args.get('user_input')
        print(some_input)
        # Case if empty
        if some_input =="":
                return render_template("index.html",my_input = some_input, my_form_result="Empty")
        else:
                render_template("index.html",my_input = some_input, my_form_result="Processing")
                #person_name, score, precetage = functions.get_usr_input(some_input, tag_words, fit_rf, stopwords)
                some_output=person_name
                some_number=score[0]
                some_image="./img/pour_wine_3.jpg"

        return render_template("index.html", my_input=some_input,my_output=some_output,my_number=some_number,my_img_name=some_image,my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
