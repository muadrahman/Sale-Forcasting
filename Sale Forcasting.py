#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# In[3]:


file_path = r"D:\CV things\ML projects\Train.csv"  # Specify the file path
dt = pd.read_csv(file_path)  # Read the CSV file into a DataFrame

# Use 'df' for further processing
print(dt.head())  # Displaying the first few rows as an example


# In[4]:


# Handle missing values
dt['Item_Weight'].fillna(dt['Item_Weight'].mean(), inplace=True)
dt['Outlet_Size'].fillna(dt['Outlet_Size'].mode()[0], inplace=True)


# In[5]:


# Feature Engineering
dt['Outlet_Years'] = 2022 - dt['Outlet_Establishment_Year']
dt['New_Item_Type'] = dt['Item_Identifier'].apply(lambda x: x[:2])
dt['New_Item_Type'] = dt['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})


# In[6]:


# Define categorical columns for encoding
cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']


# In[7]:


# Encode categorical columns
dt_encoded = pd.get_dummies(dt, columns=cat_cols)


# In[8]:


# Define features and target variable
X = dt_encoded.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = np.log1p(dt_encoded['Item_Outlet_Sales'])  # Log transformation of target variable


# In[9]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[10]:


# RandomForestRegressor
rf = RandomForestRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)

best_rf = rf_random.best_estimator_
y_pred_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R2 Score (Random Forest): {r2_rf}")

