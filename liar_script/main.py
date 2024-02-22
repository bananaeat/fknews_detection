import utils
import vectorize
import baseline
import networks

import torch
import numpy as np

liar_train, liar_test, liar_valid = utils.load_dataset('liar')
liar_train = utils.preprocess_dataset(liar_train)
liar_valid = utils.preprocess_dataset(liar_valid)

corpus_train = liar_train.iloc[:, 1].values
corpus_valid = liar_valid.iloc[:, 1].values
X_train, vectorizer = vectorize.vectorize(corpus_train)
X_valid = vectorize.vectorize_predict(corpus_valid, vectorizer)
y_train = liar_train.iloc[:, 0].values.astype(np.float32)
y_valid = liar_valid.iloc[:, 0].values.astype(np.float32)

print("Data loaded, preprocessed and vectorized")
print("Training model...")

# Logistic Rregression
# accruacy = baseline.linear_regression(X_train, y_train, X_valid, y_valid)
# print("Linear Regression, accruacy:", accruacy)

# Feedforward Network
accruacy = networks.feedforward_network(X_train, y_train, X_valid, y_valid)
print("Feedforward Network, accruacy:", accruacy)

# Decision Tree
# accruacy = baseline.decision_tree(X_train, y_train, X_valid, y_valid)
# print("Decision Tree, accruacy:", accruacy)

# Random Forest
# accruacy = baseline.random_forest(X_train, y_train, X_valid, y_valid)
# print("Random Forest, accruacy:", accruacy)

# AdaBoost
# Cross-validation
# for n_estimators in [10, 50, 100, 200, 500]:
#     for learning_rate in [0.01, 0.05, 0.1]:
#         accruacy = baseline.adaboost(X_train, y_train, X_valid, y_valid, n_estimators=n_estimators, learning_rate=learning_rate)
#         print(f"AdaBoost, n_estimators={n_estimators}, learning_rate={learning_rate}, accruacy:", accruacy)




