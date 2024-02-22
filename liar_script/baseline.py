def linear_regression(X_train, y_train, X_valid, y_valid):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import utils

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    X_train = utils.convert_csr_to_sparse_tensor(X_train)
    X_valid = utils.convert_csr_to_sparse_tensor(X_valid)

    # Define the model
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            x = self.linear(x)
            x = torch.sigmoid(x)  # Sigmoid function to squash output
            return x

    input_size = X_train.shape[1]

    model = LinearRegressionModel(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):  # number of epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        outputs = outputs.squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_valid)
        y_pred = y_pred.squeeze()
    return accuracy(y_pred, y_valid)

def decision_tree(X_train, y_train, X_valid, y_valid):
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_pred_clipped = np.clip(y_pred, 0, 1)
    return accuracy(y_pred_clipped, y_valid)

def random_forest(X_train, y_train, X_valid, y_valid):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_pred_clipped = np.clip(y_pred, 0, 1)
    return accuracy(y_pred_clipped, y_valid)

def adaboost(X_train, y_train, X_valid, y_valid, n_estimators=100, learning_rate=0.1):
    from sklearn.ensemble import AdaBoostRegressor
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return accuracy(y_pred, y_valid)

def accuracy(y_pred, y_true):
    import torch
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32)
    return torch.sum(torch.abs(y_pred - y_true) <= 0.125).item() / y_true.shape[0]