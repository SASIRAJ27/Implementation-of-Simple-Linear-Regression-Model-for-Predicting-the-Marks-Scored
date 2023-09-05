# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages
2.Assigning hours To X and Scores to Y
3.Plot the scatter plot
4.Use mse,rmse,mae formmula to find

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SASIRAJKUMAR T J
RegisterNumber:  212222230137

import numpy as np
import pandas as pd
df=pd.read_csv('student_scores.csv')

#assigning hours To X and Scores to Y
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print("X=",X)
print("Y=",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='brown')
plt.plot(X_train,reg.predict(X_train),color='orange')
plt.title('Training set(H vs S)',color='green')
plt.xlabel('Hours',color='pink')
plt.ylabel('Scores',color='pink')

plt.scatter(X_test,Y_test,color='brown')
plt.plot(X_test,reg.predict(X_test),color='violet')
plt.title('Test set(H vs S)',color='brown')
plt.xlabel('Hours',color='grey')
plt.ylabel('Scores',color='grey')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)



*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![196404656-7709c516-f896-441f-a873-9eff58768add](https://github.com/SASIRAJ27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497176/aec2899e-0d84-443c-9abd-645c19ec5803)
![196404649-da09cc70-e5fb-4301-babd-b7e90d22e16b](https://github.com/SASIRAJ27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497176/447d5341-5d25-48cc-8912-a4975dd2183e)
![196404661-88e74021-2bb8-411f-a444-9dba7fab00ee](https://github.com/SASIRAJ27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497176/eee4b095-1ecf-4c85-a162-791f09987d51)
![196404634-0406f187-78b8-4745-8485-291462cba588](https://github.com/SASIRAJ27/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497176/2d7dc9c2-589f-440f-8d0b-e9ec37751fa6)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
