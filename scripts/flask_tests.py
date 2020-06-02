from flask import Flask, render_template
app = Flask(__name__)
#to run from terminal have to set environment with: export FLASK_APP=flask_tests.py
#to execute in debug mode (dont have to reload every time there is a change)
    # run export FLASK_DEBUG=1

@app.route("/")
@app.route("/home")
def home():
    return render_template('home_test.html')

@app.route("/about")
def about():
    return render_template('about_test.html')


if __name__ == "__main__":
    app.run(debug=True)
