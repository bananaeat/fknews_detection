import pandas as pd
import nltk

def load_dataset(dataset):
    # load train, test and valid tsvs
    train = pd.read_csv(f'../data/{dataset}/train.tsv', sep='\t', header=None)
    test = pd.read_csv(f'../data/{dataset}/test.tsv', sep='\t', header=None)
    valid = pd.read_csv(f'../data/{dataset}/valid.tsv', sep='\t', header=None)

    return train, test, valid

def preprocess(text, lemmatizer=nltk.stem.WordNetLemmatizer(), stemmer=nltk.stem.PorterStemmer()):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # Remove stop-words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Remove punctuation
    text = [word for word in text if word.isalpha()]
    # Lemmatize
    text = [lemmatizer.lemmatize(word) for word in text]
    # Stemming
    text = [stemmer.stem(word) for word in text]

    return ' '.join(text)

def preprocess_label(label):
    if label == 'true':
        return 1
    elif label == 'false':
        return 0
    elif label == 'half-true':
        return 0.5
    elif label == 'mostly-true':
        return 0.75
    elif label == 'barely-true':
        return 0.25
    elif label == 'pants-fire':
        return 0

def preprocess_dataset(dataset, lemmatizer=nltk.stem.WordNetLemmatizer(), stemmer=nltk.stem.PorterStemmer()):
    dataset = dataset.copy()
    dataset.iloc[:, 2] = dataset.iloc[:, 2].apply(lambda x: preprocess(x, lemmatizer=lemmatizer, stemmer=stemmer))
    dataset.iloc[:, 1] = dataset.iloc[:, 1].apply(lambda x: preprocess_label(x))
    return dataset.iloc[:, 1:3]

def convert_csr_to_sparse_tensor(csr_matrix):
    import torch
    import numpy as np
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))