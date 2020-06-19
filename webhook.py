from flask import Flask
from flask_assistant import Assistant, ask, tell, request
import logging
import nltk
from nltk.corpus import names
import random
import semPred as sp

logging.getLogger('flask_assistant').setLevel(logging.DEBUG)
app = Flask(__name__)
assist = Assistant(app, route='/')
# begin training
sp.train()


@assist.action('greeting')
def greet_and_start():
    speech = "Hi! How are you feeling?"
    return ask(speech)


@assist.action('moodResponse')
def ask_for_mood():
    phrase = request['queryResult']['queryText']
    sem = phrase
    # sp.predict(phrase)

    msg = ''
    if sem == 'I am feeling great':
        msg = 'That is great to hear!!'
    elif sem == 'Tired':
        msg = 'Oh! I\'m sorry to hear that.'

    return ask(msg)


if __name__ == '__main__':
    app.run(debug=True)
