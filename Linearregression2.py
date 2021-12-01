import numpy as np

class LinearRegression:
    def __init__(self, w, b):
        self.w = w
        self.b = w

    def fit(self, X, y):
        print("Training...")
        X_rows, X_columns = X.shape
        y_rows = y.shape[0]
        if (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            if X_rows != y_rows:
                raise ValueError
        else:
            raise ValueError
        ones = np.ones(X_rows)
        new_X = np.insert(X, X_columns, ones, axis=1)
        transpose_X = np.transpose(new_X)
        theta = np.dot(np.linalg.inv(np.dot(transpose_X, new_X)), np.dot(transpose_X, y))
        self.w = theta[0:len(theta)-1]
        self.b = theta[-1]

    def predict(self, X):
        print("Predicting...")
        if self.w[0] == None or self.b==None:
            raise ValueError
        y = np.dot(X, self.w) + self.b
        return y

    def evaluate(self, X, y):
       print("Evaluating")
       if self.w[0] == None or self.b==None:
           raise ValueError
       try:
            y_hat = LinearRegression.predict(self,X)
            X_rows, X_columns = X.shape
            ydiff = y_hat - y
            transpose_y = np.transpose(ydiff)
            MSE = 1 / X_rows * (np.dot(transpose_y, ydiff))
            print("-> MSE = ",MSE)
            return(y_hat, MSE)
       except ValueError:
            print("*Problem with the predict function.* ")
            print("The value w or b value seems to be empty.\nCheck if fit function is called and is working properly.")

