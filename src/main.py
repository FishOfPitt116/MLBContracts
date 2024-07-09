from sklearn import svm

class Example:
    def __init__(self, a1, a2, a3):
        self.attr1 = a1
        self.attr2 = a2
        self.attr3 = a3

    def data_to_list(self):
        return [self.attr1, self.attr2, self.attr3]

e1 = Example(0, 0, 0)
e2 = Example(2, 2, 2)
X = [e1.data_to_list(), e2.data_to_list()]
y = [0.5, 2.5]
regression = svm.SVR()
regression.fit(X, y)
print(regression.predict([[1, 1, 1]]))

def contract(position, age, service_time, stats):
    pass