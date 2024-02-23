import utils
import model
from sklearn.feature_extraction.text import TfidfVectorizer

true_news, fake_news = utils.load_dataset('isot')

train_true_news, test_true_news, train_fake_news, test_fake_news = utils.train_test_split(true_news, fake_news)
train, test = utils.mix_datasets(train_true_news, test_true_news, train_fake_news, test_fake_news)

corpus_train, y_train = train.iloc[:, 1].values, train.iloc[:, 4].values
corpus_train = [utils.preprocess(text) for text in corpus_train]
corpus_test, y_test = test.iloc[:, 1].values, test.iloc[:, 4].values
corpus_test = [utils.preprocess(text) for text in corpus_test]

print("Dataset loaded and preprocessed")

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)

print("Dataset vectorized and ready to be used in the model")

# Logistic Regression
lr, accuracy = model.logistic_regression(X_train, y_train, X_test, y_test)
print(f"Logistic Regression accuracy: {accuracy}")

# SVM
svc, accuracy = model.svm_classifier(X_train, y_train, X_test, y_test)
print(f"SVM accuracy: {accuracy}")

# Random Forest
rf, accuracy = model.random_forest(X_train, y_train, X_test, y_test)
print(f"Random Forest accuracy: {accuracy}")