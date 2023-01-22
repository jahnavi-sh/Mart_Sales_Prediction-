#understanding the problem statement for the project - 
#Retail stores keep track of each individual item's sales data like item name, price, etc.. in order to meet consumer demand and update inventory management. 
#Anomalies and general trends are often discovered by mining the data warehouse's data store. For retailers, the resulting data can be used to forecast 
#future sales volume using many machine learning techniques.
#the project is to create a machine learning algorithm to study the past sales trends and make correct future sales predictions. 

#workflow for the project -  
#1. load insurance cost data 
#2. data analysis and axploration 
#3. data preprocessing 
#4. train test split 
#5. model used - XGBoost Regressor
#6. model evaluation

#import libraries 
#linear algebra - construct matrices
import numpy as np 

#data preprocessing and exploration 
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

#model training and evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics 

#load data 
big_mart_data = pd.read_csv(r'sales.csv')

#explore the data  
#view the first five rows of data 
big_mart_data.head()
#data contains the following columns 
#1. Item_Identifier - code for the identification of items
#2. Item_Weight - weight of item 
#3. Item_Fat_Content - low fat or regular 
#4. Item_Visibility - percentage of item visibility 
#5. Item_Type - category 
#6. Item_MRP - price amount of item 
#7. Outlet_Identifier - code for the identification of outlet store 
#8. Outlet_Establishment_Year
#9. Outlet_Size 
#10.Outlet_Location_Type
#11.Outlet_Type
#12.Item_Outlet_Sales

#view the total number of rows and columns 
big_mart_data.shape
#the dataset has 8523 rows (8523 data points) and 12 columns (12 features as mentioned above)

#statistical measures of the dataset
big_mart_data.describe()

#more information on the dataframe
big_mart_data.info()

#categorical features 

#fix missing values 
big_mart_data.isnull().sum()
#item_weight and outlet_size have a large number of missing values. 1463 and 2410 respectively
#all the rest of the columns have 0 missing values 

#data preprocessing 
#we will replace missing values with mean value for item_weight
#we will replace missing values with mode for outlet_size

#item_weight
big_mart_data['Item_Weight'].mean()
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
big_mart_data.isnull().sum()

#outlet_size
mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode)[0])
print (mode_of_outlet_size)
missing_values = big_mart_data['Outlet_Size'].isnull()
print (missing_values)
big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size)
big_mart_data.isnull().sum()

#view statistical measures  
big_mart_data.describe() 

#numerical features 
sns.set()

#item_weight distribution 
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()

#item_visibility distribution 
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()

#item_MRP distribution 
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()

#item_outlet_sales distribution 
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()

#item_establishment_year distribution 
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Establishment_Year', data=big_mart_data)
plt.show()

#categorical data 

#item_fat_content
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

#item_type
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

#outlet size
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

#data preprocessing 
#convert categorical value to numerical values 
big_mart_data['Item_Fat_Content'].value_counts()
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
big_mart_data['Item_Fat_Content'].value_counts()

#label encoding 
encoder = LabelEncoder()
big_mart_data['Item_Identifier']=encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content']=encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type']=encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Size']=encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Item_Location_Type']=encoder.fit_transform(big_mart_data['Item_Location_Type'])
big_mart_data['Outlet_Type']=encoder.fit_transform(big_mart_data['Outlet_Type'])

#split feature and target
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#train model 
#xgboost regressor 
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

#evaluate model 
training_data_prediction = regressor.predict(X_train)
#r squared value 
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print ('r squared value', r2_train)
#r squared error for training data is 0.63

#test data
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print ('r squared value', r2_test)
#r squared error for test data is 0.58