def begin():
    import settings
    import pandas as pd
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import settings
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from collections import defaultdict
    import gensim
    import joblib
    import numpy as np
    import settings

    path = "./"
    emoBank = "dreadditData.csv"
    #meta = "meta.tsv"
    eb = pd.read_csv(path + emoBank, index_col=0)
    #meta = pd.read_csv(path + meta, sep='\t', index_col=0)
    #eb = eb.join(meta, how='inner')

    tmp, test = train_test_split(
        eb.index, random_state=42, test_size=1000)
    train, dev = train_test_split(
        tmp, random_state=42, test_size=1000)
    relfreqs = {}
    splits = {'train': train, 'dev': dev, 'test': test}
    for key, split in splits.items():
      relfreqs[key] = eb.loc[split].value_counts() / len(split)
    pd.DataFrame(relfreqs).round(3)

    for key, split in splits.items():
      eb.loc[split, 'split'] = key

    eb.to_csv('dreadditData.csv')
    train = eb.loc[train]
    settings.test = eb.loc[test]

    train_text=train['text']
    train_V=train['label']
    test_text=settings.test['text']
    test_V=settings.test['label']

    textList = train_text.to_list()
    bowList = [doc.split() for doc in textList]
    settings.w2v = gensim.models.Word2Vec(bowList,size=350, window=10, min_count=2, iter=20)

    train_docVector = train_text.apply(document_vector)
    test_docVector = test_text.apply(document_vector)
    target = train['label']

    settings.X = list(train_docVector)
    settings.X_test = list(test_docVector)
    settings.lb = LabelEncoder()
    settings.y = settings.lb.fit_transform(target)

def document_vector(doc):
    import settings
    import numpy as np
    print("doc")
    print(doc)
    for word in doc.split():
        if(word in settings.w2v.wv.vocab):
            print(word)
    doc = [word for word in doc.split() if word in settings.w2v.wv.vocab]
    if(len(doc)==0):
      return np.zeros(350,dtype=float)
    return np.mean(settings.w2v[doc],axis=0)

def embed(natural_words):
    import settings
    import pandas as pd
    natural_word_df = pd.Series([natural_words])
    train_docVector = natural_word_df.apply(document_vector)

    phraseEmbedding = list(train_docVector)
    print("PHRASEEMBEDDINGS")
    print(phraseEmbedding)
    return phraseEmbedding

def train():
    import settings
    import pickle
    from sklearn.svm import SVC

    settings.menPred = SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    settings.menPred.fit(settings.X, list(settings.y))

    # save the model to disk
    filename = './finalized_model.sav'
    pickle.dump(settings.menPred, open(filename, 'wb'))

def test():
    import pickle
    import pandas as pd
    import settings
    from sklearn.preprocessing import LabelEncoder
    import numpy
    from sklearn.metrics import classification_report


    savedModel = './finalized_model.sav'
    menPred = pickle.load(open(savedModel,'rb'))
    y_test = menPred.predict(settings.X_test)
    y_pred = settings.lb.inverse_transform(y_test)
    print("TEST")
    print(settings.test['text'])
    test_id = [id_ for id_ in settings.test['text']]
    sub = pd.DataFrame({'id': test_id, 'label': y_pred}, columns=['id', 'label'])
    print(classification_report(settings.test['label'], list(sub['label'])))
