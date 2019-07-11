# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:05:43 2019

@author: nidhal baccouri
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


path = 'C:/Users/nidha/OneDrive/Desktop/Codes/ML Projects/datasets/linear_regression_ex1.txt'

data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

#print(data.describe())

#data.plot(kind='scatter', x='Population', y='Profit', figsize=(5,5))

m, n = data.shape
#print("m: ", m)
#print("="*50)
#print("n: ", n)

data.insert(0, 'Ones', 1)
#print("data: ", data.head())
#print("="*50)

rows, cols = m, n+1
X = data.iloc[:,0: cols-1]
y = data.iloc[:, cols-1:cols]
#print("X: ", X.head(10))
#print("y: ", y.head(10))

X = np.matrix(X)
y = np.matrix(y)
m, n = X.shape
theta = np.zeros((1,2))
theta = np.matrix(theta)
#print("X: ", X)
#print("y: ", y)
#print("X shape: ", X.shape)
print("theta shape: ", theta.shape)

def cost(X, y, theta):
    hypothesis = X * theta.T
    err = hypothesis - y
    J = sum(np.power(err, 2)) / (2*m)
    return J
    

#J = cost(X, y, theta)
#print("cost computed: ", J)    
 
def gradient_decent(X, y, theta, alpha, iters):
    temp = np.zeros(theta.shape)
    params = theta.shape[1]
    c = np.zeros(iters)
    for i in range (iters):
        err = (X * theta.T) - y
        for j in range (params):
            term = np.multiply(err, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/m)* sum(term))
        theta=temp
        c[i] = cost(X, y, theta)
        #print(c[i])
    return theta, c

alpha = 0.01
iters = 2000
g, costs = gradient_decent(X, y, theta, alpha, iters)

print("g => ", g)
print("J => ", costs)

J = cost(X, y, g)
print("optimale cost =>", J)

mn = data.Population.min()
mx = data.Population.max()
print(f"min : {mn} and max: {mx}")
x = np.linspace(mn, mx, 100)

# best fit line
f = g[0,0] + g[0,1]*x

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend()
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')



   