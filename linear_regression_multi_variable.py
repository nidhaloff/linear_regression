# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:48:34 2019

@author: nidhal baccouri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
path = 'C:/Users/nidha/OneDrive/Desktop/Codes/ML Projects/datasets/linear_regression_multi_variables.txt'
                
##read data    
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

#show data
print('data = ')
print(data2.head(10) )
print()
print('data.describe = ')
print(data2.describe())

# rescaling data
data2 = (data2 - data2.mean()) / data2.std()

print()
print('data after normalization = ')
print(data2.head(10) )


# add ones column
data2.insert(0, 'Ones', 1)


# separate X (training data) from y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]


#print('**************************************')
#print('X2 data = \n' ,X2.head(10) )
#print('y2 data = \n' ,y2.head(10) )
#print('**************************************')


# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))


#print('X2 \n',X2)
print('X2.shape = ' , X2.shape)
print('**************************************')
#print('theta2 \n',theta2)
print('theta2.shape = ' , theta2.shape)
print('**************************************')
#print('y2 \n',y2)
print('y2.shape = ' , y2.shape)
print('**************************************')


## initialize variables for learning rate and iterations
#alpha = 0.1
#iters = 100
#
## perform linear regression on the data set
#g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
#
## get the cost (error) of the model
#thiscost = computeCost(X2, y2, g2)
#
#
#print('g2 = ' , g2)
#print('cost2  = ' , cost2[0:50] )
#print('computeCost = ' , thiscost)
#print('**************************************')
#
#
## get best fit line for Size vs. Price
#
#x = np.linspace(data2.Size.min(), data2.Size.max(), 100)
#print('x \n',x)
#print('g \n',g2)
#
#f = g2[0, 0] + (g2[0, 1] * x)
#print('f \n',f)
#
## draw the line for Size vs. Price
#
#fig, ax = plt.subplots(figsize=(5,5))
#ax.plot(x, f, 'r', label='Prediction')
#ax.scatter(data2.Size, data2.Price, label='Training Data')
#ax.legend(loc=2)
#ax.set_xlabel('Size')
#ax.set_ylabel('Price')
#ax.set_title('Size vs. Price')
#
#
## get best fit line for Bedrooms vs. Price
#
#x = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
#print('x \n',x)
#print('g \n',g2)
#
#f = g2[0, 0] + (g2[0, 1] * x)
#print('f \n',f)
#
## draw the line  for Bedrooms vs. Price
#
#fig, ax = plt.subplots(figsize=(5,5))
#ax.plot(x, f, 'r', label='Prediction')
#ax.scatter(data2.Bedrooms, data2.Price, label='Traning Data')
#ax.legend(loc=2)
#ax.set_xlabel('Bedrooms')
#ax.set_ylabel('Price')
#ax.set_title('Size vs. Price')
#
#
#
## draw error graph
#
#fig, ax = plt.subplots(figsize=(5,5))
#ax.plot(np.arange(iters), cost2, 'r')
#ax.set_xlabel('Iterations')
#ax.set_ylabel('Cost')
#ax.set_title('Error vs. Training Epoch')
