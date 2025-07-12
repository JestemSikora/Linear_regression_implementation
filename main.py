import numpy as np
import matplotlib as plt
import pandas as pd
import random
from linear_regression_model import Linear_Regression

# My implementation of shuffle function.
# However, it's better to use df.sample()
def Shuffle_data_function():
    for i in range(2000):
        indx1 = random.randint(3,2937)
        temp = df.iloc[indx1].copy()
        indx2 = random.randint(3,2937)

        df.iloc[indx1] = df.iloc[indx2]
        df.iloc[indx2] = temp

    df.to_csv('Shuffled_Data.csv', index= False)
    return df

'''UWAGA: Trzeba naprawić trochę te funkcję bo czasem wychodzi poza przedział ^'''

# Data Prepering
df = pd.read_csv(r'C:\Users\wikto\OneDrive\Pulpit\Linear_Regression_implementation\Life Expectancy Data.csv')
df = Shuffle_data_function()
df.drop(['Country','Year','Status'], axis=1, inplace=True)
df = df.dropna() # Usuwanie wierszy, gdzie pokazała sie jakkiekolwiek NaN

# Splitting to Train Data and Test Data
# We can also use train_test_split function from scikit learn
# But I wanted to do this by hand :)

train_num = int(0.8 * len(df))
train_data = df.head(train_num)
Y_train = train_data['Life expectancy ']
X_train = train_data.drop('Life expectancy ', axis=1)


test_num = int(len(df) - train_num)
test_data = df.tail(test_num)
Y_test = test_data['Life expectancy ']
X_test = test_data.drop('Life expectancy ', axis=1)


model_1 = Linear_Regression()

# Training our model:
# old_value, weights, w0 = model_1.fit(X_train,Y_train, None, None)
# print(f'Stare wartości: {old_value[:5]}, {old_value[5:]}')

cost = model_1.cost_function(X_train,Y_train, None, None)
print(f"Prediction of cost {cost}")

# Gradient Descent
for i in range(20):
    new_value = model_1.gradient_descent(X_train, Y_train)

print(f'Nowe wartości: {new_value}')




