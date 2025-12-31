# Travel Reviews 

## Project Overview
This project analyzes travel review data to understand customer satisfaction using machine learning techniques.

### Dataset
This dataset was obtained from UCI Machine Learning Repository and contains customer feedback and reviews.

### Tools and Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Introduction

### Business Problem
A travel company is experiencing varying levels of satisfaction but lacks clear understanding of the key factors 
causing dissatisfaction. The company has lost resources on wrong improvements and the management is seeking a data
driven guidance on what matters most to the customers.

### Objectives
- Develop a predictive model that classifies customer reviews as Low Satisfaction or High Satisfaction.
- Identify the most influential factors affecting customer satisfaction.
- Understand how different predictors drive customer satisfaction.

### Data Loading
  We first import the necessary tools needed.
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
```
 Then load the data
 ```python
df = pd.read_csv("./tarvel+review+ratings.zip")
df.head()
```

### Data Pre-Processing
First we rename the columns for better understanding of the data 
```python
category_names = {
    "Category 1": "Art_galleries",
    "Category 2": "Dance_clubs",
    "Category 3": "Juice_bars",
    "Category 4": "Restaurants",
    "Category 5": "Museums",
    "Category 6": "Resorts",
    "Category 7": "Parks",
    "Category 8": "Beaches",
    "Category 9": "Theaters",
    "Category 10": "Religious_institutions",
    "Category 11": "Entertainment",
    "Category 12": "Value_for_Money",
    "Category 13": "Service",
    "Category 14": "Staff_Friendliness",
    "Category 15": "Language_Accessibility",
    "Category 16": "Internet",
    "Category 17": "Information",
    "Category 18": "City_Cleanliness",
    "Category 19": "Environment",
    "Category 20": "Crowd_Management",
    "Category 21": "Weather",
    "Category 22": "Navigation",
    "Category 23": "Convenience",
    "Category 24": "Overall_Travel_Satisfaction"
}
df = df.rename(columns=category_names)
df.head()
```

Then we drop the columns that are irrelevant. In this case,category 25 will be dropped .
```python
df = df.drop(columns=["Unnamed: 25"])
df.head()
```
We then check for missing values
```python
df.isna().sum()
```
Then fill in the missing values with the mean 
```python
df.fillna(df.mean(numeric_only=True), inplace=True)
df["Satisfaction"] = df["Overall_Travel_Satisfaction"].apply(
    lambda x: 1 if x >= 4 else 0
)
```
We then create a variable Satisfaction to represent Overall Travel Satisfaction and apply 1 to represent high satisfaction and 0 to
represent low satisfaction.
```python
df["Satisfaction"] = df["Overall_Travel_Satisfaction"].apply(
    lambda x: 1 if x >= 4 else 0
)
```
Then define our target and predictor variables
```python
X = df.drop(columns=["User", "Overall_Travel_Satisfaction", "Satisfaction"])
y = df["Satisfaction"]
X = X.apply(pd.to_numeric, errors="coerce")
```

#### Perform a Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```
Then counter-check for missing values in X-train and X-test
```python
X_train.isna().sum()
```
```python
X_test.isna().sum()
```
Our X-train has 1 missing value which we will fill in with the mean.
```python
X_train = X_train.fillna(X_train.mean())
```

#### Scaling the Data
To create a logistic model,we need to scale our data to normalize it.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```python
y.value_counts(normalize=True)
```

### Modelling
We buiLt two models to get the highest performance for our task.We created a logistic model and Decison tree model.

#### Logistic Regression
```python
log_reg = LogisticRegression(max_iter=1000,random_state=42)
log_reg.fit(X_train_scaled,y_train)
```
Then use logistic regression to make predictions of y.
```python
y_train_pred = log_reg.predict(X_train_scaled)
y_test_pred = log_reg.predict(X_test_scaled)
```
Then we test the accuracy,precision,recall and f1 to see how well the model works.
```python
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test,y_test_pred)
recall = recall_score(y_test,y_test_pred)
f1 = f1_score(y_test,y_test_pred)
```
```python
print(f"""Logistic Regression results:
      
Train Accuracy: {train_accuracy:.3f}
Test Accuracy:  {test_accuracy:.3f}
Precision: {precision:.3f}
Recall:    {recall:.3f}
F1-score:  {f1:.3f}
""")
```
```text
Logistic Regression results:
      
Train Accuracy: 0.912
Test Accuracy:  0.906
Precision: 0.214
Recall:    0.032
F1-score:  0.055
```

### Decision Tree
```python
dt = DecisionTreeClassifier(criterion='entropy',class_weight='balanced',max_depth=5,random_state=42)
dt.fit(X_train, y_train)
```
Then use decision tree to make predictions of y.
```python
y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)
print("Decision tree y_train value =",y_train_pred_dt)
print("Decision tree y_test value =",y_test_pred_dt)
```
Then we test the accuracy,precision,recall and f1 to see how well the model works.
```python
train_accuracy = accuracy_score(y_train, y_train_pred_dt)
test_accuracy = accuracy_score(y_test, y_test_pred_dt)
precision = precision_score(y_test,y_test_pred_dt)
recall = recall_score(y_test,y_test_pred_dt)
f1 = f1_score(y_test,y_test_pred_dt)
```
```python
print(f"""Decision Tree results:
           
Train Accuracy: {train_accuracy:.3f}
Test Accuracy:  {test_accuracy:.3f}
Precision: {precision:.3f}
Recall:    {recall:.3f}
F1-score:  {f1:.3f}
""")
```
```text
Decision Tree results:
      
Train Accuracy: 0.830
Test Accuracy:  0.813
Precision: 0.307
Recall:    0.916
F1-score:  0.460
```


### Evaluation
Logistic Regression and Decision tree models were trained and evaluated.
Decision Tree showed a higher recall score and F1-score and was therefore selected for predicting customer satisfaction.
We then sort the X values to see the most influential predictors in predicting customer satisfaction.
```python
feature_importance = pd.Series(dt.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_importance)
```
The most influential predictors were Theatres and Language Accessiblity, indicating that ratings in these categories 
play a major role in predicting traveller satisfaction.

#### Visualizing the Travel Review Decision Tree
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
y_pred = dt.predict(X_test)
plt.figure(figsize=(12,8)) 
plot_tree(dt,
    feature_names=X.columns,
    class_names=['Low Satisfaction','High Satisfaction'],
    filled=True,
    rounded=True,
    fontsize=10  
)
plt.title("Decision Tree - Travel Reviews ")
plt.show
```
## 📈 Decision Tree

![Decision Tree](./Decision%20tree.png)

The decision tree reveals how predictor variables interact to determine customer satisfaction.Reviews equal to 2 represents
high satisfaction while  reviews below 1 represent low satisfaction.Theatres is the most important predictor and influenced
the best reviews. Other factors that influenced best reviews include Language accessibility and art galleries. Environment,
weather and beaches got low reviews thus low satisfaction from the clients.The decision tree is interpretable and allows the
business to understand why a customer is predicted to be satisfied or not. 

## Key Findings
- Decision tree achieved highest recall and f1 score compared to the other model and was therefore used to make predictions on
  customer satisfaction
- Top splits,theatres, indicate highest satisfaction.

## Recommendation
- Improve service quality in theatres and art galleries for higher reviews
- Advice the tourists to visit when there are favourable weather conditions
- Make improvements on beaches,museums and the environment for better reviews





























