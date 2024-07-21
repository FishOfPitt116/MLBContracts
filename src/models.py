from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def linear_regression(data, x, y):
    X = data[x]
    Y = data[y]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=116)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # # The coefficients
    # print("Coefficients: \n", model.coef_)

    # MSE - closer to 0, better predictor
    print("Mean Squared Error: %.2f" % mean_squared_error(Y_test, Y_pred))
    
    # R2 - between 0 and 1 where closer to 1 is better
    print("R-Squared Score: %.2f" % r2_score(Y_test, Y_pred))

    return {
        "mse" : mean_squared_error(Y_test, Y_pred),
        "r2" : r2_score(Y_test, Y_pred)
    }


# from sklearn import svm

# class Example:
#     def __init__(self, a1, a2, a3):
#         self.attr1 = a1
#         self.attr2 = a2
#         self.attr3 = a3

#     def data_to_list(self):
#         return [self.attr1, self.attr2, self.attr3]

# e1 = Example(0, 0, 0)
# e2 = Example(2, 2, 2)
# X = [e1.data_to_list(), e2.data_to_list()]
# y = [0.5, 2.5]
# regression = svm.SVR()
# regression.fit(X, y)
# print(regression.predict([[1, 1, 1]]))

# def contract(position, age, service_time, stats):
#     pass

