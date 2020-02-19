import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression


N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
fit_intercept=True


y=X[2]

X[4]=X[2]


LR = LinearRegression(fit_intercept=fit_intercept)
LR.fit_vectorised(X, y,n_iter=1000)
LR.plot_contour(X,y)
