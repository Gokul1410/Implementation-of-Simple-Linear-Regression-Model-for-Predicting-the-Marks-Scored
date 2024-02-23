# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gokul C
RegisterNumber: 212223240040
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('/exp1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot 2024-02-23 102341](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/9498cdd9-d9f1-410f-a185-d177fdf51e3b)
![image](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/8a42d789-cb5f-49c9-9150-8405520b1b9d)
![image](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/7e86fb16-f847-4c28-93ee-b1b8d2ae12d4)
![image](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/975ebf51-1fda-4b18-8bc0-0a7e00d12a86)
![image](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/a21f5e23-6dbc-4a39-a84c-f5be3139dcfd)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
