# Importing libraries
from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initializing a new Flask instance
app = Flask(__name__)

# Using the route decorator
@app.route('/')

# home function
def home():
	return render_template('home.html')

# Routing to the prediction page using the route decorator
@app.route('/predict',methods=['POST'])

# predict function
def predict():
	# Reading the .csv file and converting the string from UTF-8 to Latin-1
	spam = pd.read_csv("spam.csv", encoding="latin-1")
	# Creating a new column label and mapping ham as 0 and spam as 1
	spam['label'] = spam['class'].map({'ham': 0, 'spam': 1})
	X = spam['message']
	Y = spam['label']
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	# Fit the Data
	X = cv.fit_transform(X)
	# Split the dataset into test and train
	from sklearn.model_selection import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
	# Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,Y_train)
	clf.score(X_test,Y_test)

	# Transferring the form data to the server
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

# Activating the Flask debugger
if __name__ == '__main__':
	app.run(debug=True)
