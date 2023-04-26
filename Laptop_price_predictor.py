# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:53:32 2023

@author: dhair
"""
#*********** Import The Libraries **********

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score  , mean_absolute_error

from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor ,AdaBoostRegressor , ExtraTreesRegressor
from sklearn.svm import SVR

#****************** Import Dataest ******************

laptop_df = pd.read_csv('/Users/dhair/OneDrive/Desktop/laptop_price.csv' , encoding_errors='replace')
print(laptop_df)

#print(laptop_df.info())
#print(laptop_df.isnull().sum())
#************** Drop The column ***********

laptop_df.drop(columns = ['laptop_ID'] , inplace = True)

#print(laptop_df)
#************* FeatureEngineering Drop GB and KG *********

laptop_df['Ram'] = laptop_df['Ram'].str.replace('GB','')
laptop_df['Weight'] = laptop_df['Weight'].str.replace('kg','')
#print(laptop_df)

#************* Reasign the Type of column which are related************

laptop_df['Ram'] = laptop_df['Ram'].astype('int32')
laptop_df['Weight'] = laptop_df['Weight'].astype('float32')

#print(laptop_df)
#print(laptop_df.info())

#from forex_python.converter import CurrencyRates
#c = CurrencyRates()
#laptop_df['Price_euros'] = laptop_df["Price_euros"].replace({"INR": "*90", "EUR": "*1"}, regex=True).map(pd.eval)
#laptop_df['Price_euros'] = laptop_df['Price_euros'].c.get_rate('EUR' , 'INR')

#******* plot the Dist plot respect to price ********

sns.distplot(laptop_df['Price_euros'])
plt.show()

#********* Plot the bar plot of Company related to laptop **************

laptop_df['Company'].value_counts().plot(kind = 'bar')
plt.show()
#plt.figure(figsize = (15 , 6))
sns.barplot(data = laptop_df,x = 'Company' , y = 'Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

#************* Plot the barplot on the TypeName resprct to price ************

laptop_df['TypeName'].value_counts().plot(kind = 'bar')
sns.barplot(data = laptop_df,x = 'TypeName' , y = 'Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

#******** Plot tghe distplot respect to Inches ***********

sns.distplot(laptop_df['Inches'])
plt.show()
sns.scatterplot(data = laptop_df, x = 'Inches' , y = 'Price_euros')
plt.show()

print(laptop_df['ScreenResolution'].value_counts())

#************ We are doing FeatureEngineering here **************

#***********Create a NEW column with the help of ScreenResolution *****************

laptop_df['Touchscreen'] = laptop_df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

#*********Plot the barplot on TouchScreen Laptops *******

laptop_df['Touchscreen'].value_counts().plot(kind = 'bar')
plt.show()

#**********Create another New column with the help of ScreenResolution**********
 
laptop_df['Ips'] = laptop_df['ScreenResolution'].apply(lambda x:1 if 'IPS Panel' in x else 0)
sns.barplot(x = laptop_df['Ips'], y =laptop_df['Price_euros'])
plt.show()

#********Create a BEW DataFrame with the help of ScreenResolutiion************

new = laptop_df['ScreenResolution'].str.split('x' , n = 1 , expand = True)
#print(new)

#*********** Assign the value of resolution and create NEW columns  to X and Y *******
 
laptop_df['X_res'] = new[0]
laptop_df['Y_res'] = new[1]

#**********Apply Regular Expression here and remove then unnecessary data from the X_res column ***************
 
laptop_df['X_res'] = laptop_df['X_res'].str.replace(',' , '').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

#******** Assign the Type to the data which are belong to *************

laptop_df['X_res'] = laptop_df['X_res'].astype('int')
laptop_df['Y_res'] = laptop_df['Y_res'].astype('int')
print(laptop_df.info())

#************ Now finding the correlation with respect to price***********

print(laptop_df.corr()['Price_euros']) 

#************ Create another New Column of PPI and apply PPI formula with the help of X ,Y res ,Inches *********

laptop_df['ppi'] = (((laptop_df['X_res']**2) + (laptop_df['Y_res']**2))**0.5/laptop_df['Inches']).astype('float')

print(laptop_df.corr()['Price_euros']) 

#************ Now dropout the columns ********

laptop_df.drop(columns = ['ScreenResolution' , 'Inches' , 'X_res' , 'Y_res'] , inplace = True)
print(laptop_df)

#*********** Create another NEW column with the help of CPU column in dataset***************

laptop_df['CpuName'] = laptop_df['Cpu'].apply(lambda x: " ".join(x.split() [0:3]))

#********** Create a FUNCTION to finding out the text in Given Data************


def fetch_processor (text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            
            return 'Other Intel Processor'
        else:
            return 'AMD processor'

#********* Find the text which are usefull for us in the CpuName column and Create New column***************

laptop_df['Cpu brand'] = laptop_df['CpuName'].apply(fetch_processor)
print(laptop_df)

#*********** Now plot the barplot on cpubrand and the price********

laptop_df['Cpu brand'].value_counts().plot(kind = 'bar')
plt.show()

sns.barplot(data = laptop_df , x = 'Cpu brand' , y = 'Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

#*********** Now DropOut the columns ************

laptop_df.drop(columns = ['Cpu' , 'CpuName'] , inplace = True)

#*********** Plot a barplot on Ram and price


laptop_df['Ram'].value_counts().plot(kind = 'bar')
plt.show()
sns.barplot(data = laptop_df , x = 'Ram' , y = 'Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

#*********** Now doing some more featureengineering on the column of Memory ***********

#print(laptop_df['Memory'].value_counts())

#************* Replace the .0  with nothing using Regular Expression Fun****************  

laptop_df['Memory'] = laptop_df['Memory'].astype('str').replace('\.0' , '' , regex = True)
#print(laptop_df['Memory'].value_counts())

#************* Remove the GB and TB in the memory column **************

laptop_df['Memory'] = laptop_df['Memory'].str.replace('GB' , '')
laptop_df['Memory'] = laptop_df['Memory'].str.replace('TB' , '000')

#*********** Create a NEW DataFrame with the help of Memory Column *************

new = laptop_df['Memory'].str.split('+' , n = 1 , expand = True)
#print(new)

#************* Now create Two Another column in Dataset with the of NewDataFrame ***********************

laptop_df['First'] = new[0]
laptop_df['First'] = laptop_df['First'].str.strip()
#print(laptop_df['First'])
#print(laptop_df['Memory'].value_counts())

laptop_df['Second'] = new[1]
#print(laptop_df['Second'])

#*************** Now creatw another columns with First Column **************

laptop_df['1HDD']           = laptop_df['First'].apply(lambda x : 1 if 'HDD' in x else 0)
#print('1HDD : ' , laptop_df['1HDD'])
laptop_df['1SSD']           = laptop_df['First'].apply(lambda x : 1 if 'SSD' in x else 0)
#print('1SSD : ' , laptop_df['1SSD'])
laptop_df['1Hybrid']        = laptop_df['First'].apply(lambda x : 1 if 'Hybrid' in x else 0)
#print('1Hybrid : ' , laptop_df['1Hybrid'])
laptop_df['1Flash_storage'] = laptop_df['First'].apply(lambda x : 1 if 'Flash Storage' in x else 0)
#print('1Flash Storeage : ' , laptop_df['1Flash_storage'])

#*********** Replace the string values with nothing *************

laptop_df['First'] = laptop_df['First'].str.replace(r'\D' , '')
#print(laptop_df['First'])

#********** Replace the NONE with 0 ************

laptop_df['Second'].fillna('0' , inplace = True)
#print(laptop_df['Second'])

#*************** Now create another columns with Second Column **************

laptop_df['2HDD']           = laptop_df['Second'].apply(lambda x : 1 if 'HDD' in x else 0)
#print('1HDD : ' , laptop_df['1HDD'])
laptop_df['2SSD']           = laptop_df['Second'].apply(lambda x : 1 if 'SSD' in x else 0)
#print('1SSD : ' , laptop_df['1SSD'])
laptop_df['2Hybrid']        = laptop_df['Second'].apply(lambda x : 1 if 'Hybrid' in x else 0)
#print('1Hybrid : ' , laptop_df['1Hybrid'])
laptop_df['2Flash_storage'] = laptop_df['Second'].apply(lambda x : 1 if 'Flash Storage' in x else 0)
#print('1Flash Storeage : ' , laptop_df['1Flash_storage'])

#*********** Replace the string values with nothing *************

laptop_df['Second'] = laptop_df['Second'].str.replace(r'\D' , '')

#*******  Assign the TYPE to the First and Second Column *************

laptop_df['First']  = laptop_df['First'].astype('int')
laptop_df['Second'] = laptop_df['Second'].astype('int')

#************ Now Create some more new columns with the help of First and second column *********************\
    
laptop_df['HDD']           = (laptop_df['First']*laptop_df['1HDD'] + laptop_df['Second']*laptop_df['2HDD'])
laptop_df['SSD']           = (laptop_df['First']*laptop_df['1SSD'] + laptop_df['Second']*laptop_df['2SSD'])
laptop_df['Hybrid']        = (laptop_df['First']*laptop_df['1Hybrid'] + laptop_df['Second']*laptop_df['2Hybrid'])
laptop_df['Flash_Storage'] = (laptop_df['First']*laptop_df['1Flash_storage'] + laptop_df['Second']*laptop_df['2Flash_storage'])

#*************** Now DropOut the columns *************

laptop_df.drop(columns = [ 'Memory' , 'First' , 'Second' , '1HDD' , '1SSD' , '1Hybrid' , '1Flash_storage' , '2HDD' , '2SSD' , '2Hybrid' , '2Flash_storage'] , inplace = True)

#********** Now Finding the correlation with Respect to price ***********

print(laptop_df.corr()['Price_euros'])

#************** Now Droping Out of columns ************
laptop_df.drop(columns = ['Hybrid' , 'Flash_Storage'] , inplace = True)

#****** Doing some feature engineering on the GPU column ***********

#print(laptop_df['Gpu'].value_counts())

#********** Now extract the Branding in column of GPU *************

laptop_df['GPU_Brand'] = laptop_df['Gpu'].apply(lambda x:x.split()[0] )
#print(laptop_df['GPU_Brand'].value_counts())

laptop_df = laptop_df[laptop_df['GPU_Brand'] != 'ARM']
#print(laptop_df['GPU_Brand'].value_counts())

#*********** Analysis Brand on the Price **********

sns.barplot(data = laptop_df , x = 'GPU_Brand' , y = 'Price_euros' , estimator = np.median)
plt.xticks(rotation = 'vertical')
plt.show()

#********** Dropout The Gpu column ******* 

laptop_df.drop(columns = ['Gpu'] , inplace = True)

#*********** FeatureEngineering On OS column **************
 
print(laptop_df['OpSys'].value_counts())

sns.barplot(data = laptop_df , x = 'OpSys' , y = 'Price_euros')
plt.xticks(rotation = 'vertical')
plt.show()

#*********  Now create a New Function for finding the categories of OS ********

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 10' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'MacOS'
    else:
        return 'Other/No OS/ Linux'

laptop_df['OS'] = laptop_df['OpSys'].apply(cat_os)
#print(laptop_df['OS'])

laptop_df.drop(columns = ['OpSys' , 'Product'] , inplace = True)

#*********** Now doing FeatureEngineering on Weight column **********

sns.scatterplot(data = laptop_df , x = 'Weight' , y = 'Price_euros')
plt.show()

#*********** Now check the correlation of all columns /************

sns.heatmap(laptop_df.corr() , annot = True)
plt.show()

#************* Doing FeatureEngineering of Target Column ******* 

X = laptop_df.drop(columns = ['Price_euros'])
y = np.log(laptop_df['Price_euros'])
#print(X)
#print(y)

#********* Apply Train Test Split ******************

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 2)

#print(X_train)

#********** Covert categorical columns ************


#from xgboost import XGBRegressor

#********* Apply Column Transformer  ************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = LinearRegression()

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('Linear Regressor :  ')
print('R2Score :' , r2_score(y_test, y_pred))
print('Mean Squre Error :' , mean_absolute_error(y_test, y_pred))

#************** Apply Ridge **********

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = Ridge(alpha  = 10)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('Ridge : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#*********** Apply Lasso ****************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = Lasso(alpha = 0.001)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('Lasso : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#********** Apply KNeighborsRegressor *****************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = KNeighborsRegressor(n_neighbors = 3)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('KNeighbours Regressor :')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#***************Apply DecisionTree *****************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = DecisionTreeRegressor(max_depth = 8)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('DecisionTreeRegressor : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#************ Apply SVM *************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = SVR(kernel = 'rbf' , C = 10000 , epsilon = 0.1)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('SVMRegressor : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#*********** Apply RandomForest ******************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = RandomForestRegressor(n_estimators=100 , random_state=3 , bootstrap= True ,max_samples=0.5 , max_features=0.75 , max_depth = 15)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('RandomForestRegressor : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

# ******************* Apply ExtratreeRegressor *********************

step1 = ColumnTransformer(transformers   = [('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = ExtraTreesRegressor(n_estimators = 100 , random_state=3 ,bootstrap= True, max_samples=0.5 , max_features=0.75 , max_depth = 15)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('ExtratreeRegressor : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))

#************* Apply AdaBoost ************************

step1 = ColumnTransformer(transformers =[('col_tnf' , OneHotEncoder(sparse = False , drop  = 'first'),[0 , 1 , 7 , 10 , 11])] , remainder = 'passthrough')

step2 = AdaBoostRegressor(n_estimators= 15 , learning_rate=0.1)

pipe  = Pipeline([('step1' , step1) , ('step2' , step2)])
pipe.fit(X_train , y_train)

y_pred = pipe.predict(X_test)

print('AdaBoostRegressor : ')
print('R2Score : ' , r2_score(y_test, y_pred))
print('Mean Squre Error : ' , mean_absolute_error(y_test, y_pred))


#************* Now Exporting The Model ************ 
import pickle
pickle.dump(laptop_df , open('laptop_df.pkl' , 'wb'))
pickle.dump(pipe , open('pipe.pkl' , 'wb'))






























