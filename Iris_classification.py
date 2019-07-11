# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

try:
    
    data = pd.read_csv('C:/Users/nidha/OneDrive/Desktop/Codes/ML Projects/datasets/iris.data.txt',
                       sep=" ", header=None, delimiter= ','
                       )
except:
    print("error occured")

m, n = data.shape
data = np.array(data).reshape( (m,n))    
print(data)




print(f"m: {m} and n: {n}")

X = data[:, 0:n-1]
y = data[:, n-1:]
print("X: ", X)
print("y: ", y)




    
