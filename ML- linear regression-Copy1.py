#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[44]:


dataset=pd.read_csv('insurance_data.csv')
dataset


# In[45]:


x = dataset.iloc[:, :-1].values
x


# In[46]:


y = dataset.iloc[:, -1].values
y


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# In[48]:


print(x_train)


# In[49]:


print(y_train)


# In[50]:


print(x_test)


# In[51]:


print(y_test)


# In[52]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[57]:


y_pred = regressor.predict(x_test)
y_pred


# In[58]:


new_data_point = [[30]]  # Features for the new data point
y_pre = regressor.predict(new_data_point)
print(y_pre)


# In[59]:


y_pre1 = regressor.predict([[33]])
y_pre1


# In[60]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.show()


# In[61]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.show()


# In[62]:


from sklearn.metrics import r2_score
accuracy = r2_score(y_pred,y_test)
accuracy


# In[ ]:




