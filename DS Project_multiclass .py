# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:59:21 2022

@author: arunk
"""

##Multiple Linear Regression
#lets laod the required libraries

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


#lets import the data
sample = pd.read_excel("C:/Users/arunk/OneDrive/Desktop/DS Project/final_data.xlsx")
sample.info()
sample.head()

sample.columns


## checking null values
sample.isna().sum()  # no null values

#lets remove the not required variables
sample = sample.drop(['Unnamed: 0', 'Latitudes and Longitudes (Patient)','Latitudes and Longitudes (Agent)','Latitudes and Longitudes (Diagnostic Center)'], axis=1)
sample = sample.drop(['PatientID'], axis=1)
sample = sample.drop(['Time_slot'], axis=1)
sample = sample.drop(['pincode'], axis=1)
sample = sample.drop(['Test_Booking_Date','Test_Booking_Time _HH:MM','Sample_Collection_Date'], axis=1)
sample.describe()


## rename the column
sample = sample.rename(columns={'ExactArrivalTime ':'ExactArrivalTime'})
sample = sample.rename(columns={'AgentArrivalTime ':'AgentArrivalTime'})
sample = sample.rename(columns={'Availabilty_time ':'Availabilty_time'})

## checking duplicates
duplicate = sample.duplicated()
duplicate
sum(duplicate) # no duplicates found

from sklearn.preprocessing import RobustScaler, LabelEncoder
#Label encoding for categorical features
cat_col = ["patient_location", "DiagnosticCenters", "Gender", "Test_name", "Sample", "Way_Of_Storage_Of_Sample","Availabilty_time","AgentArrivalTime"]

lab = LabelEncoder()
mapping_dict ={}
for col in cat_col:
    sample[col] = lab.fit_transform(sample[col])
 
    le_name_mapping = dict(zip(lab.classes_,
                        lab.transform(lab.classes_))) #To find the mapping while encoding
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)

sample.columns


##From description we can see that Test_Booking_Date is a object data type,\ Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction

##For this we require pandas to_datetime to convert object data type to datetime dtype.

sample["Test_Booking_day"] = pd.to_datetime(sample.Test_Booking_Date, format="%d/%m/%Y").dt.day
sample["Test_Booking_month"] = pd.to_datetime(sample["Test_Booking_Date"], format = "%d/%m/%Y").dt.month


# Since we have converted Test_Booking_Date column into integers, Now we can drop as it is of no use.

sample.drop(["Test_Booking_Date"], axis = 1, inplace = True)


# Test_Booking_Time _HH:MM is when a plane leaves the gate. 

# Extracting Hours
sample["Test_Booking_hour"] = pd.to_datetime(sample["Test_Booking_Time _HH:MM"]).dt.hour

# Extracting Minutes
sample["Test_Booking_min"] = pd.to_datetime(sample["Test_Booking_Time _HH:MM"]).dt.minute

# Now we can drop Dep_Time as it is of no use
sample.drop(["Test_Booking_Time _HH:MM"], axis = 1, inplace = True)



##From description we can see that Sample_Collection_Date is a object data type,\ Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction

##For this we require pandas to_datetime to convert object data type to datetime dtype.

sample["Sample_Collection_day"] = pd.to_datetime(sample.Sample_Collection_Date, format="%d/%m/%Y").dt.day
sample["Sample_Collection_month"] = pd.to_datetime(sample["Sample_Collection_Date"], format = "%d/%m/%Y").dt.month


# Since we have converted Test_Booking_Date column into integers, Now we can drop as it is of no use.

sample.drop(["Sample_Collection_Date"], axis = 1, inplace = True)


sample.AgentArrivalTime = sample.AgentArrivalTime.astype(object)
sample.AgentArrivalTime = sample.AgentArrivalTime.astype('float')
# AgentArrivalTime  is when a plane leaves the gate. 

sample.columns


#Seaborn is a library for making statistical graphics and Seaborn helps to explore and understand the data
import seaborn as sns 
# Jointplot :  a plot of two variables with bivariate and univariate graphs. 
# This function provides a convenient interface to the ‘JointGrid’ class, with several canned plot kinds
sns.jointplot(x=sample['shortest_distance_Agent_Pathlab'], y=sample['ExactArrivalTime'])
sns.jointplot(x=sample['shortest_distance_Patient_Pathlab'], y=sample['ExactArrivalTime'])
sns.jointplot(x=sample['shortest_distance_Patient_Agent'], y=sample['ExactArrivalTime'])


# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(sample['shortest_distance_Agent_Pathlab'])
sns.countplot(sample['shortest_distance_Patient_Pathlab'])
sns.countplot(sample['shortest_distance_Patient_Agent'])

# Scatter plot between the variables along with histograms
import seaborn as sns #for Advanced visualizations
sns.pairplot(sample.iloc[:, :])                          
# Correlation matrix 
sample.corr()

sample.columns
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('ExactArrivalTime ~ Test_Booking_day + Test_Booking_hour + Sample_Collection_day', data = sample).fit() # regression model
ml1

# Summary
ml1.summary()

# p-values for Test_Booking_day, Test_Booking_hour , Sample_Collection_dayare more than 0.05
# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row
sample_new = sample.drop(sample.index[[623]])
sample_new

# Preparing model                  
ml_new = smf.ols('ExactArrivalTime ~ Test_Booking_day + Test_Booking_hour + Sample_Collection_day', data = sample_new).fit()    
# Summary
ml_new.summary()


sample.columns
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd = smf.ols('patient_location ~ shortest_distance_Agent_Pathlab + shortest_distance_Patient_Pathlab', data = sample).fit().rsquared  
vif_rd = 1/(1 - rsq_rd) 

rsq_ms = smf.ols('shortest_distance_Agent_Pathlab ~ shortest_distance_Patient_Pathlab+ patient_location', data = sample).fit().rsquared  
vif_ms = 1/(1 - rsq_ms)

rsq_ad = smf.ols('shortest_distance_Patient_Pathlab ~ shortest_distance_Agent_Pathlab+ patient_location', data = sample).fit().rsquared  
vif_ad = 1/(1 - rsq_ad) 


# Storing vif values in a data frame
d1 = {'Variables':['patient_location','shortest_distance_Agent_Pathlab','shortest_distance_Patient_Pathlab'], 'VIF':[vif_rd, vif_ms, vif_ad]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# As shortest_distance_Agent_Pathlab is having highest VIF value, we are going to drop this from the prediction model
# Final model
final_ml = smf.ols('ExactArrivalTime ~ patient_location + shortest_distance_Patient_Pathlab', data = sample).fit()
final_ml.summary() 

import matplotlib.pyplot as plt 
# Prediction
pred = final_ml.predict(sample)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = sample.ExactArrivalTime, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()
sm.graphics.influence_plot(final_ml)


sample = sample.drop(['AgentArrivalTime'], axis=1)
sample = sample.drop(['Availabilty_time'], axis=1)




X = sample.iloc[:,:14]
X


Y = sample.iloc[:, 15]
Y


# Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y , test_size = 0.2) # 20% test data

#now lets build a model using fitting Multilinear Regression to the training set
from sklearn.linear_model import LinearRegression
sample_f = LinearRegression() 
sample_f.fit(X_train, Y_train)



#prediction for the test results
y_pred = sample_f.predict(X_test)
y_pred

# Finding the MAE, MSE, RMSE socre 
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


#for comarision to check whether the y_test is good or y_pred is good
#we will use r^2 score from the sklearn package
from sklearn.metrics import r2_score
score = r2_score(Y_test, y_pred) 
score # 0.6782269954685929

print('Train Score: ', sample_f.score(X_train, Y_train))  #0.6222936412476994
print('Test Score: ', sample_f.score(X_test, Y_test))     #0.6782269954685929

# saving the model
# importing pickle
import pickle
pickle.dump(sample_f, open('sample_f.pkl', 'wb'))

# load the model from
model = pickle.load(open('sample_f.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(sample.iloc[:,:14])
list_value

print(model.predict(list_value))














