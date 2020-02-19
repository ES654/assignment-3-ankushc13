import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from metrics import *
import pandas as pd

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

# A= np.power(x,3)
# A[np.isnan(A)] = 0
# print(pd.DataFrame(A))

fit_intercept=True

theta1=[]
degree=[]

for i in range(1,15):
    poly = PolynomialFeatures(i)
    X=poly.transform(x)
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_normal(pd.DataFrame(X),pd.Series(y)) # here you can use fit_non_vectorised / fit_autograd methods
    theta1.append(LR.print_theta(X,y))
    degree.append(i)
    # y_hat = LR.predict(X)
    # print('RMSE: ', rmse(y_hat, y))
    # print('MAE: ', mae(y_hat, y))


print(theta1)

plt.plot(degree,theta1)
plt.xlabel("degree")
plt.ylabel("theta")
plt.show()