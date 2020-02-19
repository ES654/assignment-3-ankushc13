import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
import sympy

np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
LR = LinearRegression(fit_intercept=True)





print("Vectorised")
for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    LR.plot_line_fit(X,y)
    LR.plot_contour(X,y)
    LR.plot_surface(X,y)