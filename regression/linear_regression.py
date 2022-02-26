#Linear Regression
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Gathering the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
## Separating the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
## Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
## Predict the test dataset result
y_pred = regressor.predict(X_test)
## Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title of the training set')
plt.xlabel('X-label title')
plt.ylabel('Y-label title')
plt.show()
## visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test set title')
plt.xlabel('X-label title')
plt.ylabel('Y-label title')
plt.show()