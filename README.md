# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import Libraries: Load necessary libraries (Pandas, Scikit-learn).

2.Load Data: Read the employee data from a CSV file.

3.Explore Data: Display dataset information and check for missing values.

4.Preprocess Data: Encode categorical variables and define features (X) and target (y).

5.Split Data: Divide the data into training (80%) and testing (20%) sets.

6.Train Model: Initialize and fit a Decision Tree Classifier on the training data.

7.Make Predictions: Predict employee churn on the test set.

8.Evaluate Model: Calculate and display the model's accuracy.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PRAVEEN K
RegisterNumber: 212223230153
*/
```
```python
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/RANJEETH17/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120718823/47e2a472-50b5-47be-8063-5b9a919b0423)
![image](https://github.com/RANJEETH17/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120718823/53dc33ec-6238-47ae-a106-05352df257ea)
![image](https://github.com/RANJEETH17/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120718823/1dafbcf9-fbaf-469c-8858-f63ae3893928)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
