#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df1=pd.read_csv("Best_Movies_Netflix.csv")


# In[4]:


df1


# In[5]:


df1.isnull().sum()


# In[6]:


ind_train=df1.iloc[:,2:3]
dep_train=df1.iloc[:,3:6]


# In[7]:


ind_train


# In[8]:


dep_train


# In[9]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(ind_train,dep_train,test_size=0.20)


# In[10]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression()
lnr.fit(xtrain,ytrain)


# In[11]:


predicting=lnr.predict(xtest)
predicting


# In[12]:


xtest


# In[13]:


ytest


# In[14]:


plt.scatter(xtest,ytest["SCORE"],color='r',label="Year-Wise Score")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Score")


# In[15]:


plt.scatter(xtest,ytest["NUMBER_OF_VOTES"],color='b',label="Year-Wise Votes")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Number of Votes")


# In[16]:


plt.scatter(xtest,ytest["DURATION"],color='b',label="Year-Wise Duration")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Duration")


# In[16]:


df2=df1=pd.read_csv("Best Shows Netflix.csv")


# In[17]:


df2


# In[18]:


df2.isnull().sum()


# In[19]:


ind_train_show=df2.iloc[:,2:3]
dep_train_show=df2.iloc[:,3:7]


# In[20]:


ind_train_show


# In[21]:


dep_train_show


# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(ind_train_show,dep_train_show,test_size=0.20)


# In[23]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression()
lnr.fit(xtrain,ytrain)


# In[24]:


predicting2=lnr.predict(xtest)
predicting2


# In[25]:


xtest


# In[26]:


ytest


# In[27]:


plt.scatter(xtest,ytest["SCORE"],color='r',label="Year-Wise Score of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Score")


# In[28]:


plt.scatter(xtest,ytest["NUMBER_OF_VOTES"],color='r',label="Year-Wise Votes for Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Votes")


# In[29]:


plt.scatter(xtest,ytest["DURATION"],color='r',label="Year-Wise Duration of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Duration")


# In[30]:


plt.scatter(xtest,ytest["NUMBER_OF_SEASONS"],color='r',label="Year-Wise Seasons of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("No. of Seasons")


# In[ ]:




