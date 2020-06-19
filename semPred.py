def train():
    import settings
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
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

    settings.init()
    eb = pd.read_csv('emobank.csv', index_col=0)
    meta = pd.read_csv('meta.tsv', sep='\t', index_col=0)
    eb = eb.join(meta, how='inner')

    tmp, test = train_test_split(
        eb.index, stratify=eb.category, random_state=42, test_size=1000)
    train, dev = train_test_split(
        tmp, stratify=eb.loc[tmp].category, random_state=42, test_size=1000)

    relfreqs = {}
    splits = {'train': train, 'dev': dev, 'test': test}
    for key, split in splits.items():
        relfreqs[key] = eb.loc[split].category.value_counts() / len(split)
    pd.DataFrame(relfreqs).round(3)

    for key, split in splits.items():
        eb.loc[split, 'split'] = key

    eb = eb.drop(columns=['document', 'category', 'subcategory'])

    eb.to_csv('emobank.csv')

    valance = eb['V']
    text = eb['text']

    sentiment = []
    for v in valance:
        if(v >= 3.5):
            sentiment.append('pos')
        else:
            sentiment.append('neg')
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        text,
        sentiment,
        train_size=0.80,
        random_state=1234)

    #result = cross_val_score(etree_w2v_tfidf, X_trainS, y_trainS,  cv=5)
    result = settings.genPredictor.fit(X_trainS, y_trainS)
    # print(result.mean())


def predict(phrase):
    settings.genPredictor.predict([phrase])
    return prediction
