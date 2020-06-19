
class TfidfEmbeddingVectorizer(object):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 6

    def fit(self, X, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        import numpy as np
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def init():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    global genPredictor

    with open("glove.6B.50d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}

    genPredictor = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("ExtraTrees", ExtraTreesClassifier())])
