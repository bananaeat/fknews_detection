from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer