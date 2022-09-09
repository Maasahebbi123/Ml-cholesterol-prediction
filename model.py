#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('BMI_Data.csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df=df.drop(['Date','Unnamed: 4','Unnamed: 5'],axis=1)
df.head()


# In[5]:


df.describe()


# In[6]:


df.corr()


# In[7]:


df.info()


# In[8]:


import seaborn as sns


# In[9]:


sns.heatmap(df.corr(),cmap='YlGnBu',annot=True)
plt.show()


# In[10]:


c_w=df.plot.scatter(x='Weight in Pounds',y='Cholesterol')


# So from above scatter plot we can observe that as weight increases Cholestro level goes on increasing.And animals with weight more than 100% pounds will have more cholestrol level.

# In[11]:


W_out=sns.boxplot(df['Weight in Pounds'])


# In[12]:


C_B=df.plot.scatter(x='BMI',y='Cholesterol')


# In[13]:


W_out=sns.boxplot(df['BMI'])


# From above scatter plot we can see that Animals with BMI 84-90 have high cholestrol level.

# In[14]:


X=df.drop(['Cholesterol'],axis=1)


# In[15]:


Y=df['Cholesterol']


# In[16]:


print(X)


# In[17]:


print(Y)


# # Splitting data into train and test data

# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[19]:


X.shape,X_train.shape,X_test.shape


# In[20]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# # Modeling
Random Forest Regressor
# In[21]:


regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, Y_train)


# In[22]:


Y_pred1=regr.predict(X_test)


# In[23]:


from sklearn.metrics import mean_squared_error, r2_score


# In[24]:


import sklearn.metrics as metrics


# In[26]:


Score_1=metrics.mean_squared_error(Y_test,Y_pred1)
np.sqrt(Score_1)


# In[27]:


Score_11=metrics.r2_score(Y_test,Y_pred1)
Score_11

LINEAR REGRESSION
# In[28]:


reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred2=reg.predict(X_test)


# In[30]:


Score2=metrics.mean_squared_error(Y_test,Y_pred2)
np.sqrt(Score2)


# In[72]:


Score22=metrics.r2_score(Y_test,Y_pred2)
Score22

xg BOOST
# In[31]:


import xgboost as xg


# In[32]:


xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)


# In[33]:


xgb_r.fit(X_train,Y_train)
Y_pred3 = xgb_r.predict(X_test)
Score3=metrics.mean_absolute_error(Y_test,Y_pred3)
Score3


# In[34]:


Score33=metrics.mean_squared_error(Y_test,Y_pred3)
np.sqrt(Score33)


# In[36]:


import pickle


# In[38]:


pickle.dump(regr, open("model.pkl", "wb"))


# In[ ]:




