from flask import Flask
from flask_assistant import Assistant, ask, tell, request
import logging
import random
import pickle5 as pickle
import settings
import ml

logging.getLogger('flask_assistant').setLevel(logging.DEBUG)
app = Flask(__name__)
assist = Assistant(app, route='/')

savedModel = './finalized_model.sav'
menPred = pickle.load(open(savedModel, 'rb'))

settings.init()
ml.begin()
ml.train()
ml.test()

@assist.action('greeting')
def greet_and_start():
    speech = "Hi! How are you feeling?"
    return ask(speech)


@assist.action('moodResponse')
def ask_for_mood():
    phrase = request['queryResult']['queryText']
    print("Model")
    print(menPred)
    print("PHRASE")
    print(phrase)
    print(type(menPred))
    prediction = settings.menPred.predict(ml.embed(phrase))[0]

    print("PREDICTION")
    print(prediction)
    msg = ''
    if prediction == 0:
        msg = 'That is great to hear!!'
    elif prediction == 1:
        msg = 'Oh! I\'m sorry to hear that.'

    return ask(msg)

if __name__ == '__main__':
    app.run(debug=True)
