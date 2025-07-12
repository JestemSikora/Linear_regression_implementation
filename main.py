import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import random
from linear_regression_model import Linear_Regression
from sklearn.preprocessing import StandardScaler

# My implementation of shuffle function

def Shuffle_data_function():
    for i in range(2000):
        indx1 = random.randint(3,2937)
        temp = df.iloc[indx1].copy()
        indx2 = random.randint(3,2937)

        df.iloc[indx1] = df.iloc[indx2]
        df.iloc[indx2] = temp

    df.to_csv('Shuffled_Data.csv', index= False)
    return df

def StandardScaler_fun(X):

    # Mean from each feature
    average_feature = []
    std = []
    for i in range(X.shape[1]):
        sum = 0
        for j in range(X.shape[0]):
            sum += X.iloc[j, i]
            average_feature.append(sum / X.shape[0])
        std.append(np.std(X.iloc[:,i]))

    # Actutual equation
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            X.iloc[j, i] = (X.iloc[j, i] - average_feature[i]) / std[i]
            

    return X

'''UWAGA: Trzeba naprawić trochę te funkcję bo czasem wychodzi poza przedział ^'''

# Data Prepering
df = pd.read_csv(r'C:\Users\wikto\OneDrive\Pulpit\Linear_Regression_implementation\Life Expectancy Data.csv')
df = Shuffle_data_function()
df.drop(['Country','Year','Status'], axis=1, inplace=True)
df = df.dropna() 

# Splitting to Train Data and Test Data
train_num = int(0.8 * len(df))
train_data = df.head(train_num)
Y_train = train_data['Life expectancy ']
X_train = train_data.drop('Life expectancy ', axis=1)
X_train = StandardScaler_fun(X_train)


test_num = int(len(df) - train_num)
test_data = df.tail(test_num)
Y_test = test_data['Life expectancy ']
X_test = test_data.drop('Life expectancy ', axis=1)
X_test = StandardScaler_fun(X_test)

# Plotting actual data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.scatter(X_pca[:,0], X_pca[:,1], c=Y_train, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of n-dimensional data')
plt.colorbar(label='Target Class')

# plt.show()

model_1 = Linear_Regression()

# Training our model:
# old_value, weights, w0 = model_1.fit(X_train,Y_train, None, None)
# print(f'Stare wartości: {old_value[:5]}, {old_value[5:]}')

cost = model_1.cost_function(X_train,Y_train, None, None)
print(f"Prediction of cost {cost[0]}")

# Gradient Descent
for i in range(80):
    new_value = model_1.gradient_descent(X_train, Y_train, cost[1], cost[2])

print(f'Nowe wartości: {new_value}')





