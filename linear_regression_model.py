import numpy as np
import random

class Linear_Regression:
    def _init_(self,X,y):
        self.X = X
        self.y = y
        self.m = X.shape[0]

    def start_weights(self, X, y):
        # Random weights   
        weights = []
        weights_0 = round(random.uniform(-2, 2), 2)
        for i in range(len(X.iloc[1])):
            weights.append(round(random.uniform(0, 2), 2))

        return weights, weights_0

    def fit(self, X, y, weights, w0):
        y_prediction = []
        if weights is None and w0 is None:
            weights, w0 = self.start_weights(X,y)

        # Linear Regression Equation
        for i in range(X.shape[0]):
            y_prediction.append(X.iloc[i] @ weights + w0)
        #print(f'Length of y: {len(y_prediction)}\n, Wiersze: {X.shape[0]},\n y: {y.shape[0]}')

        return y_prediction, weights, w0
    
    def cost_function(self, X, y, weights, w0):
        y_prediction, weights, w0 = self.fit(X, y, None, None)
        # print(y_prediction)
        #print(f'Length of y: {len(y_prediction)}\n, Wiersze: {X.shape[0]},\n y: {y.shape[0]}')
        m = X.shape[0]
        cost = 0
        gap = 0
        for j in range(m):
            gap += (y_prediction[j] - y.iloc[j])**2
        cost = 1/(2*m) * gap
            
        return cost
        
    def gradient_descent(self, X, y):
        alfa = 0.01
        y_prediction, weights, w0 = self.fit(X,y,None, None)
        sum_w = 0
        sum_w0 = 0
        list_cost_function = []

        for i in range(len(weights)):
            for j in range(X.shape[0]):
                sum_w0 += (y_prediction[j] - y.iloc[j])
                sum_w += (y_prediction[j] - y.iloc[j])*X.iloc[j]
            
            w0 = w0 - alfa*1/(X.shape[0]) * sum_w0
            weights[i] = weights[i] - alfa * 1/(X.shape[0]) * sum_w

            
        list_cost_function = self.cost_function(X, y, weights, w0)


        return list_cost_function
        

