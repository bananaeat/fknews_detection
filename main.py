import utils
import vectorize
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

liar_train, liar_test, liar_valid = utils.load_dataset('liar')
liar_train = utils.preprocess_dataset(liar_train)
liar_valid = utils.preprocess_dataset(liar_valid)

corpus_train = liar_train.iloc[:, 1].values
corpus_valid = liar_valid.iloc[:, 1].values
X_train, vectorizer = vectorize.vectorize(corpus_train)
X_valid = vectorizer.transform(corpus_valid)
y_train = liar_train.iloc[:, 0].values
y_valid = liar_valid.iloc[:, 0].values

print("Data loaded, preprocessed and vectorized")
print("Training model...")

# Linear Rregression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
loss = sklearn.metrics.mean_squared_error(y_valid, y_pred)

print("Linear Regression, Mean Squared loss:", loss)

# Decision Tree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
loss = sklearn.metrics.mean_squared_error(y_valid, y_pred)

print("Decision Tree, Mean Squared loss:", loss)

# Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
loss = sklearn.metrics.mean_squared_error(y_valid, y_pred)

print("Random Forest, Mean Squared loss:", loss)

# AdaBoost
model = AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
loss = sklearn.metrics.mean_squared_error(y_valid, y_pred)

print("AdaBoost, Mean Squared loss:", loss)




