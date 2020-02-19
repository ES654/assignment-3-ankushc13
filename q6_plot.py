import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from metrics import *
import pandas as pd



x = np.array([i*np.pi/180 for i in range(60,1260,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

# A= np.power(x,3)
# A[np.isnan(A)] = 0
# print(pd.DataFrame(A))

fit_intercept=True



for j in range(100,300,90):
    theta1=[]
    degree=[]
    for i in [1,3,5,7,9]:
        poly = PolynomialFeatures(i)
        X=poly.transform(x)
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_normal(pd.DataFrame(X[:j]),pd.Series(y[:j])) # here you can use fit_non_vectorised / fit_autograd methods
        theta1.append(LR.print_theta(X,y))
        degree.append(i)
    print(theta1)
    plt.scatter(degree,theta1,label="N = "+str(j))
    plt.legend(prop={'size': 6},borderpad=2)

plt.xlabel("degree")
plt.ylabel("theta")
plt.show()
    # y_hat = LR.predict(X)
    # print('RMSE: ', rmse(y_hat, y))
    # print('MAE: ', mae(y_hat, y))



