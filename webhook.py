from flask import Flask
from flask_assistant import Assistant, ask, tell
import logging
import nltk
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import names
import random

nltk.download('names')
logging.getLogger('flask_assistant').setLevel(logging.DEBUG)

#collect male and female names
m = names.words('male.txt')
f = names.words('female.txt')

#shuffle the set
random.seed(1234) # Set the random seed to allow replicability
names = ([(name,'male') for name in m] + [(name,'female') for name in f])
random.shuffle(names)

#Convert word to one-hot character
def one_hot_character(c):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = [0]*(len(alphabet)+1)
    i = alphabet.find(c.lower())
    if i >= 0:
        result[i] = 1
    else:
        result[len(alphabet)] = 1 # character is out of the alphabet
    return result

def sk_gender_features5(word):
    "Return the one-hot encoding of the last 5 characters"
    features = []
    for i in range(5):
        if i <= len(word):
            features += one_hot_character(word[-i])
        else:
            features += one_hot_character(' ')
    return features

#Partition the sets
sk_train_set5=[(sk_gender_features5(n),g) for (n,g) in train_names]
sk_devtest_set5=[(sk_gender_features5(n),g) for (n,g) in devtest_names]
sk_classifier5=MultinomialNB()
train5_X, train5_y = zip(*sk_train_set5)

#Train the classifier
sk_classifier5.fit(train5_X, train5_y)
devtest5_X, devtest5_y = zip(*sk_devtest_set5)

app = Flask(__name__)
assist = Assistant(app, route='/')


@assist.action('greeting')
def greet_and_start():
    speech = "Hey! What is your name?"
    return ask(speech)

@assist.action('giveName', mapping={'name': 'sys.given-name'})
def ask_for_color(name):
    print("THE NAME IS ")
    print(name)
    gender = sk_classifier5.classify(gender_features5(name))

    if gender == 'male':
        gender_msg = 'Hi Mr {}!'.format(name)
    else:
        gender_msg = 'Hi Miss {}!'.format(name)

    speech = gender_msg + ' What is your favorite color?'
    return ask(speech)

if __name__ == '__main__':
    app.run(debug=True)