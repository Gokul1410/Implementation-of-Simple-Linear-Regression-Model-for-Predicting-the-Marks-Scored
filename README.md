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
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1 (1).csv')
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
![Screenshot 2024-02-23 105650](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/6b57ba1a-2777-44ad-a597-cbec2f013813)
![Screenshot 2024-02-23 105828](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/3e1c0586-cb48-47c6-a1d0-9fc7714a5124)
![Screenshot 2024-02-23 105923](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/e1f1fb74-78df-4926-a6e6-4dd3b8885b82)
![Screenshot 2024-02-23 110008](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/f4a70f6f-d09e-47ca-896f-28f71feab996)
![Screenshot 2024-02-23 102341](https://github.com/Gokul1410/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153058321/d6b243d8-7564-4764-bccf-8980341eabbb)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
