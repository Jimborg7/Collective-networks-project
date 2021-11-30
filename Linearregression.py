import numpy as np


class LinearRegression:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def fit(self, X, y):
        X_rows, X_columns = X.shape
        print("Work like a charm")
        print(X.shape)

        y_rows = y.shape[0]
        print(X_rows, y_rows)
        if (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            print("We made it through")
            if X_rows != y_rows:
                raise ValueError("The number of rows of the two arrays are not the same.")
        else:
            raise ValueError("All WRoNG")
        ones = np.ones(X_rows)
        new_X = np.insert(X, X_columns, ones, axis=1)
        inverseX = np.linalg.inv(new_X)
        theta = np.dot(np.dot(inverseX, new_X) ^ (-1), np.dot(inverseX, y))
        for i in range(theta - 1):
            self.w.append(theta[i])
        self.b = theta[-1]

    def predict(self, X):
        if self.w.isempty():
            raise ValueError("The value w seems to be empty. Use the fit function first.")
        y = np.dot(X, self.w) + self.b
        return y

    def evaluate(self, X, y):
        if self.w.isempty():
            raise ValueError("The value w seems to be empty. Use the fit function first.")
        y_hat = LinearRegression.predict(X)
        X_rows, X_columns = X.shape
        ydiff = y_hat - y
        inverse_y = np.linalg.inv(ydiff)
        MSE = 1 / X_rows * (np.dot(inverse_y, ydiff))
        return (y_hat, MSE)
