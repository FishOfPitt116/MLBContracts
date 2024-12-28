from joblib import dump, load
import os
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def linear_regression(data, x, y):
    X = data[x]
    Y = data[y]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=116)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return {
        "mse" : mean_squared_error(Y_test, Y_pred),
        "r2" : r2_score(Y_test, Y_pred),
        "model" : model
    }

def support_vector_regression(data, x, y):
    X = data[x]
    Y = data[y]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=116)
    model = SVR()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return {
        "mse" : mean_squared_error(Y_test, Y_pred),
        "r2" : r2_score(Y_test, Y_pred),
        "model" : model
    }

def lasso_regression(data, x, y, alpha=0.1):
    X = data[x]
    Y = data[y]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=116)
    model = Lasso(alpha=alpha)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return {
        "mse" : mean_squared_error(Y_test, Y_pred),
        "r2" : r2_score(Y_test, Y_pred),
        "model" : model
    }

def write_model_to_file(model, filename):
    dump(model, filename)

def read_model_from_file(filename):
    if not os.path.exists(filename):
        return None
    return load(filename)