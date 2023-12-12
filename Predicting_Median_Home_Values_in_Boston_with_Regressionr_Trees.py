#%% Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.tree as tree

#%% Dataset Import
data = pd.read_csv('real_estate_data.csv')
print(data.head())
print("Data Shape:{}", data.shape)
print("Number of empty data\n", data.isna().sum())

# %% Pre-Processing
data.dropna(inplace=True)
print("Number of empty data\n", data.isna().sum())
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

print("X DATA SET \n", X.head())
print("\n Y DATA SET \n", Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 2, random_state=1)

#%% Regression Tree

#Creat Model
regression_tree = DecisionTreeRegressor(criterion= "squared_error")

#Training
regression_tree.fit(X_train,Y_train)
#Prediction
prediction = regression_tree.predict(X_test)
print("$", (prediction - Y_test).abs().mean()*1000)

#Evaluation
print("Score:",regression_tree.score(X_test,Y_test))

