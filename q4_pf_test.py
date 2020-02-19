import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd


x = np.array([i*np.pi/180 for i in range(60,100,4)])


poly = PolynomialFeatures(2)
print("before transformation\n ",pd.DataFrame(x))
print("before transformation\n ",pd.DataFrame(poly.transform(x)))
