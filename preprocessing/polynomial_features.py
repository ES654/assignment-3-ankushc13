''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree=degree
        self.include_bias=include_bias
        
        pass

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        Xtr=X.copy()
        if self.include_bias==True:
            if self.degree>1:
                for i in range(2,self.degree+1):
                    b= np.power(X,i)
                    Xtr=np.column_stack([Xtr,b])
                bia=np.ones(X.shape[0])
                Xtr = np.column_stack([bia,Xtr]) 
        else:
            if self.degree>1:
                for i in range(2,self.degree+1):
                    b= np.power(X,i)
                    Xtr=np.column_stack([Xtr,b])
        return Xtr
        pass
    
        
        
        
        
        
        
        
        
    
                
                
