from sklearn import datasets
from sklearn import model_selection
from linearregression import LinearRegression
import numpy as np

boston = datasets.load_boston()
w = None
b = None
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(boston.data,boston.target, train_size=0.7, shuffle=False)
lr = LinearRegression(w, b)
try:
    lr.fit(X_train, y_train)
except ValueError:
    print("*Problem with the fit function:")
    print("1)Either the X,y valus are not numpy arrays or\n2)the given arrays have different number of rows,therefor are not compatible. ")
try :
    Y_MSE = lr.evaluate(X_test,y_test)
    RMSE = np.sqrt(Y_MSE[1])
    print("=> RMSE = ", RMSE)
except ValueError:
    print("Problem with the evaluation function.\nValues for w or b haven't been given.")
    print("Training required: use fit function and check if w or b are given values.")
