from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def vectorize_predict(corpus, vectorizer):
    X = vectorizer.transform(corpus)
    return X

def nonnegative_matrix_factorization(corpus, n_components=10, init='random', random_state=0):
    from sklearn.decomposition import NMF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    return W, H, vectorizer, model

def nonnegative_matrix_factorization_predict(corpus, vectorizer, model):
    X = vectorizer.transform(corpus)
    W = model.transform(X)
    H = model.components_
    return W, H