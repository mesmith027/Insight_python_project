from flask import Flask, render_template
app = Flask(__name__)
#to run from terminal have to set environment with: export FLASK_APP=flask_tests.py
#to execute in debug mode (dont have to reload every time there is a change)
    # run export FLASK_DEBUG=1

posts =[
    {
        'author': 'M.E.Smith',
        'title': 'Blog 1',
        'content': 'First post content',
        'date_posted': 'June 3 2020'
    },
    {
        'author': 'J. Doe',
        'title': 'Blog 2',
        'content': 'second post content',
        'date_posted': 'June 4 2020'
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home_test.html', posts=posts)

@app.route("/about")
def about():
    return render_template('about_test.html')


if __name__ == "__main__":
    app.run(debug=True)
