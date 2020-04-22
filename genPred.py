import nltk
nltk.download('names')
from nltk.corpus import names
import random

#collect male and female names
m = names.words('male.txt')
f = names.words('female.txt')

#shuffle the set
random.seed(1234) # Set the random seed to allow replicability
names = ([(name,'male') for name in m] + [(name,'female') for name in f])
random.shuffle(names)

#partition the set
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

#use suffixes up to 5 in size to train onto the dataset
def gender_features5(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:],
            'suffix3': word[-3:],
            'suffix4': word[-4:],
            'suffix5': word[-5:]}
train_set5=[(gender_features5(n),g) for (n,g) in train_names]
devtest_set5=[(gender_features5(n),g) for (n,g) in devtest_names]
classifier5=nltk.NaiveBayesClassifier.train(train_set5)
nltk.classify.accuracy(classifier5,devtest_set5) 
print(classifier5.classify(gender_features5('Tracy')))