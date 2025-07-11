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
        weights_0 = round(random.uniform(0, 4), 2)
        for i in range(len(X.iloc[1])):
            weights.append(round(random.uniform(0, 4), 2))

        return weights, weights_0

    def fit(self, X, y, weights, w0):

        if weights is None and w0 is None:
            weights, w0 = self.start_weights(X,y)

        # Linear Regression Equation
        for i in range(int(w0)):
            y_prediction = X.iloc[i] @ weights + w0
            print(f'Prediction of y: {y_prediction}')

        return y_prediction, weights, w0
    
    def cost_function(self, X, y, weights, w0):
        y_prediction = self.fit(X, y, None, None)
        m = X.shape[0]
        cost = 0
        for j in range(m):
            cost += 1/(2*m)*(y_prediction[0] - y.iloc[j])**2

        return cost
        
    def gradient_descent(self, X, y):
        alfa = 0.01
        y_prediction, weights, w0 = self.fit(X,y,None, None)
        sum_w = 0
        sum_w0 = 0
        list_cost_function = []

        for i in range(len(weights)):
            for j in range(len(X.shape[0])):
                sum_w0 += (y_prediction[j] - y[j])
                sum_w += (y_prediction[j] - y[j])*X[i]
            
            w0 = w0[i] - alfa*1/(self.m) * sum_w0
            weights = weights[i] - alfa * 1/(self.m) * sum_w

            
            list_cost_function.append(self.cost_function(X,y,weights,w0))


        return list_cost_function
        

