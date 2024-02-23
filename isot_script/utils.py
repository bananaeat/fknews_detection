import pandas as pd
import nltk

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

def load_dataset(dataset):
    # load train, test and valid tsvs
    true_news = pd.read_csv(f'../data/{dataset}/True.csv', sep=',', header=None)
    fake_news = pd.read_csv(f'../data/{dataset}/Fake.csv', sep=',', header=None)

    return true_news, fake_news

def train_test_split(true_news, fake_news):
    # Split the data into train and test
    train_true_news = true_news.sample(frac=0.8)
    test_true_news = true_news.drop(train_true_news.index)

    train_fake_news = fake_news.sample(frac=0.8)
    test_fake_news = fake_news.drop(train_fake_news.index)

    return train_true_news, test_true_news, train_fake_news, test_fake_news

def mix_datasets(train_true_news, test_true_news, train_fake_news, test_fake_news):
    # Add labels
    train_true_news[4] = 1
    train_fake_news[4] = 0
    test_true_news[4] = 1
    test_fake_news[4] = 0

    # Mix the datasets
    train = pd.concat([train_true_news, train_fake_news])
    test = pd.concat([test_true_news, test_fake_news])

    # Shuffle the datasets
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test