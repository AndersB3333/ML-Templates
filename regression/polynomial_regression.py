#Multiple Linear Regression
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
## Training the Polynomial Regression model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
## Visualising the dataset results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(X_train), color = 'blue')
plt.title('Title of the training set')
plt.xlabel('X-label title')
plt.ylabel('Y-label title')
plt.show()
