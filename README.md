# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:25008235 
RegisterNumber: Mohamed Aathil M
*/


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee]

<img width="511" height="215" alt="Screenshot 2025-12-10 143310" src="https://github.com/user-attachments/assets/929c2a2f-0377-45f1-9797-8f7e2c6649c3" />


<img width="954" height="254" alt="Screenshot 2025-12-10 143603" src="https://github.com/user-attachments/assets/9ad61a22-d55d-4331-b57c-e2b0dd6affb1" />


<img width="319" height="107" alt="Screenshot 2025-12-10 143613" src="https://github.com/user-attachments/assets/7c7f462c-db7f-4f76-bdba-7b5af9d1a78e" />


<img width="391" height="223" alt="Screenshot 2025-12-10 143620" src="https://github.com/user-attachments/assets/250dce67-1527-460c-9fcf-77bb2291d360" />

<img width="190" height="52" alt="Screenshot 2025-12-10 143629" src="https://github.com/user-attachments/assets/a0ddf09c-fe99-4e02-930b-acad6cc2fe52" />

<img width="322" height="36" alt="Screenshot 2025-12-10 143636" src="https://github.com/user-attachments/assets/97bf2225-62dd-45b0-96a7-e0d5aadf926c" />

<img width="289" height="36" alt="Screenshot 2025-12-10 143645" src="https://github.com/user-attachments/assets/54ea72e1-ee26-47dc-9b76-7fbb12a68c6f" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
