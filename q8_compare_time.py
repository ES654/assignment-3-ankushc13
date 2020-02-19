
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
import time

np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
LR = LinearRegression(fit_intercept=True)




 
print("Vectorised \n")
for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_vectorised(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    print(end-start)

print("\n\n\n\nNon-Vectorised \n")
for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_non_vectorised(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    print(end-start)


print("\n\n\n\nAutoGrad \n")
for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_autograd(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    print(end-start)


print("\n\n\n\nNormal \n")
for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    print(end-start)