#!/usr/bin/env python
# coding: utf-8

# # E-commerce sales data reveals patterns and trends that help businesses identify opportunities for growth. By analyzing this data, companies can optimize their strategies, such as offering targeted promotions or stocking popular products. This leads to increased sales and hidden profits.

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os


# # Import the Data

# In[20]:


# Load the datasets

cloud_warehouse = pd.read_csv("Cloud Warehouse Compersion Chart.csv")
expense_iigf = pd.read_csv("Expense IIGF.csv")
international_sales = pd.read_csv("International sale Report.csv")
PL_March = pd.read_csv("PL March 2021.csv")
May_2022 = pd.read_csv("May-2022.csv")
sale_report = pd.read_csv("Sale Report.csv")
amazon_sales = pd.read_csv("Amazon Sale Report.csv")


# # Data Synopsis

# In[21]:


# Display the first few rows of each dataset
amazon_sales.head()


# In[22]:


cloud_warehouse.head()


# In[23]:


expense_iigf.head()


# In[24]:


international_sales.head()


# In[25]:


PL_March.head()


# In[26]:


May_2022.head()


# In[27]:


sale_report.head()


# # Data Quality Improvement

# In[33]:


#Exclude irrelevant columns

amazon_sales.drop(columns=['Unnamed: 22'], inplace=True, errors='ignore')
cloud_warehouse.drop(columns=['Unnamed: 1'], inplace=True, errors='ignore')
expense_iigf.drop(columns=['Unnamed: 3'], inplace=True, errors='ignore')


# # Exploratory Data Analysis

# In[34]:


#Standardize date columns to datetime

amazon_sales['Date'] = pd.to_datetime(amazon_sales['Date'], errors='coerce')
international_sales['DATE'] = pd.to_datetime(international_sales['DATE'], errors='coerce')


# In[35]:


# Visualize sales performance over time

plt.figure(figsize=(12, 6))
amazon_sales.groupby('Date')['Amount'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()


# # Sales Segmentation by Category

# In[36]:


# Depict sales trends by category
plt.figure(figsize=(12, 6))
amazon_sales.groupby('Category')['Amount'].sum().sort_values().plot(kind='barh')
plt.title('Total Sales by Category')
plt.xlabel('Total Sales')
plt.ylabel('Category')
plt.show()


# # Heatmap for Correlation analysis

# In[37]:


# Heatmap of correlations among numeric columns
numeric_df = amazon_sales.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# # Predictive Analysis

# In[38]:


# Transform data for predictive analytics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Filter features and target variable
features = ['Qty', 'ship-postal-code']
target = 'Amount'

# Eliminate rows with missing entries in specified columns
amazon_sales.dropna(subset=features + [target], inplace=True)


# Split the data
X = amazon_sales[features]
y = amazon_sales[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


# In[ ]:




