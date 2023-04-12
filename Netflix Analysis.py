#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df1=pd.read_csv("Best_Movies_Netflix.csv")


# In[6]:


df1


# In[7]:


df1.isnull().sum()


# In[8]:


ind_train=df1.iloc[:,2:3]
dep_train=df1.iloc[:,3:6]


# In[9]:


ind_train


# In[10]:


dep_train


# In[11]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(ind_train,dep_train,test_size=0.20)
xtrain.shape, ytrain.shape, xtest.shape,ytest.shape


# In[181]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression()
lnr.fit(xtrain,ytrain)


# In[182]:


predicting=lnr.predict(xtest)
predicting


# In[183]:


xtest


# In[184]:


ytest


# In[185]:


plt.scatter(xtest,ytest["SCORE"],color='r',label="Year-Wise Score")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Score")


# In[186]:


plt.scatter(xtest,ytest["NUMBER_OF_VOTES"],color='b',label="Year-Wise Votes")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Number of Votes")


# In[187]:


plt.scatter(xtest,ytest["DURATION"],color='b',label="Year-Wise Duration")
plt.plot(xtest,predicting,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Duration")


# In[188]:


df2=pd.read_csv("Best Shows Netflix.csv")


# In[189]:


df2


# In[190]:


df2.isnull().sum()


# In[191]:


ind_train_show=df2.iloc[:,2:3]
dep_train_show=df2.iloc[:,3:7]


# In[192]:


ind_train_show


# In[193]:


dep_train_show


# In[194]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(ind_train_show,dep_train_show,test_size=0.20)


# In[195]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression()
lnr.fit(xtrain,ytrain)


# In[196]:


predicting2=lnr.predict(xtest)
predicting2


# In[197]:


xtest


# In[198]:


ytest


# In[199]:


plt.scatter(xtest,ytest["SCORE"],color='r',label="Year-Wise Score of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Score")


# In[200]:


plt.scatter(xtest,ytest["NUMBER_OF_VOTES"],color='r',label="Year-Wise Votes for Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Votes")


# In[201]:


plt.scatter(xtest,ytest["DURATION"],color='r',label="Year-Wise Duration of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("Duration")


# In[202]:


plt.scatter(xtest,ytest["NUMBER_OF_SEASONS"],color='r',label="Year-Wise Seasons of Shows")
plt.plot(xtest,predicting2,color='g',linestyle="-")
plt.legend(loc="upper right")
plt.title("Linear Regression")
plt.xlabel("Year")
plt.ylabel("No. of Seasons")


# # EDA

# In[203]:


df1.describe()


# In[204]:


df2.describe()


# In[205]:


df1.columns


# In[206]:


df1.RELEASE_YEAR.count()


# In[207]:


df1.SCORE.nunique()


# In[208]:


df1["SCORE"].value_counts()


# In[209]:


np.corrcoef(df1.RELEASE_YEAR,df1.SCORE)


# In[210]:


df1.corr()


# In[211]:


df2.corr()


# In[212]:


from scipy import stats


# In[213]:


p_c =stats.pearsonr(df1.RELEASE_YEAR,df1.SCORE)
p_c


# In[214]:


import matplotlib.pyplot as plt
plt.scatter(df1.RELEASE_YEAR,df1.SCORE)


# In[215]:


plt.scatter(df1.RELEASE_YEAR,df1.NUMBER_OF_VOTES)
plt.scatter(df2.RELEASE_YEAR,df2.NUMBER_OF_VOTES)


# In[216]:


df2


# In[217]:


from pandas_profiling import ProfileReport


# In[218]:


rp =ProfileReport(df1)
rp


# # Fitting In The Decision Tree Regression

# In[60]:


df1


# In[61]:


ind_train=df1.iloc[:,5:6]
dep_train=df1.iloc[:,3:4]


# In[62]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(ind_train,dep_train,test_size=0.20)
xtrain.shape, ytrain.shape, xtest.shape,ytest.shape


# In[ ]:





# In[ ]:





# In[63]:


from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
ind_train = X_labelencoder.fit_transform(ind_train)
print (ind_train)


y_labelencoder = LabelEncoder ()
dep_train = y_labelencoder.fit_transform (dep_train)
print (dep_train)


# In[64]:


dep_train.shape,ind_train.shape


# In[65]:


from sklearn.tree import DecisionTreeClassifier
d_t= DecisionTreeClassifier(criterion ="gini",max_depth =6)


# In[66]:


xtrain


# In[67]:


d_t.fit(ind_train.reshape(-1,1),dep_train.reshape(-1,1))


# In[68]:


prd =d_t.predict(ind_train.reshape(-1,1))
prd.shape


# In[69]:


from sklearn.metrics import accuracy_score
score = accuracy_score(dep_train,prd)
score


# In[70]:


from sklearn.metrics import classification_report
print(classification_report(dep_train.reshape(-1,1),prd))


# In[77]:


from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

print("Root Mean Square Error:")
print(np.sqrt(mean_squared_error(dep_train,prd)))


# In[78]:


data_feature_name =["SCORE"]

from sklearn.tree import export_graphviz


# In[79]:


from sklearn import tree
from graphviz import Source


# In[80]:


from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(d_t,out_file =None, feature_names =data_feature_name,filled  = True,rounded =True,))
display(SVG(graph.pipe(format='svg')))


# # K-means clustering

# In[215]:


iv = df1[["SCORE","MAIN_GENRE","MAIN_PRODUCTION"]]


# In[216]:


iv


# In[246]:


iv = iv.values


# In[247]:


iv


# In[248]:


plt.figure(figsize = (10,5))
c = np.random.randint(50,220)
colormap = plt.cm.get_cmap('Reds')
for SCORE,MAIN_GENRE,MAIN_PRODUCTION in iv:
    
   
    plt.text(SCORE+.01,MAIN_GENRE,MAIN_PRODUCTION,fontdict = {'size':8})
    plt.scatter(SCORE,MAIN_GENRE, c = [c], vmin=50,vmax = 220, cmap=colormap)

plt.show()


# In[249]:


from sklearn.preprocessing import LabelEncoder


# In[250]:


lable_2 = LabelEncoder().fit(iv[:,2])
lable_1 = LabelEncoder().fit(iv[:,1])


# In[251]:


iv[:,2] = lable_2.transform(iv[:,2])
iv[:,1] = lable_1.transform(iv[:,1])


iv = iv.astype(float)


# In[252]:


from sklearn.cluster import KMeans


# In[253]:


kmeans = KMeans(n_clusters = 2 , n_init =10 , random_state =0 )


# In[254]:


kmeans.fit(iv)
kmeans.predict(iv)


# In[255]:


wss= []
for i in range(1,11) :
    kmeans = KMeans(n_clusters = i, n_init =10 ,random_state =0 )
    kmeans.fit(iv)
    wss.append(kmeans.inertia_) 
    print (i, kmeans.inertia_)


# In[256]:


plt.plot(range(1,11),wss)
plt.title("The Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.show()


# In[267]:


iv = df1[["SCORE","MAIN_GENRE","MAIN_PRODUCTION"]]
iv


# In[268]:


lable_2 = LabelEncoder().fit(iv.iloc[:,2])
lable_1 = LabelEncoder().fit(iv.iloc[:,1])


# In[269]:


iv.iloc[:,2] = lable_2.transform(iv.iloc[:,2])
iv.iloc[:,1] = lable_1.transform(iv.iloc[:,1])


iv = iv.astype(float)


# In[270]:


iv.iloc[:,2] ,iv.iloc[:,1] ,iv


# In[271]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5 , n_init =10 , random_state =0 )
kmeans.fit_predict(iv)


# In[272]:


iv['cluster']=kmeans.fit_predict(iv)


# In[273]:


iv


# In[274]:


from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_score_average = silhouette_score(iv, kmeans.fit_predict(iv))
print(silhouette_score_average)


# In[283]:


plt.scatter(iv.loc[iv['cluster']==0,'SCORE'],iv.loc[iv['cluster']==0,'MAIN_GENRE'],s=100,c='red',label ="sports")
plt.scatter(iv.loc[iv['cluster']==1,'SCORE'],iv.loc[iv['cluster']==1,'MAIN_GENRE'],s=100,c='green',label= "animation")
plt.scatter(iv.loc[iv['cluster']==2,'SCORE'],iv.loc[iv['cluster']==2,'MAIN_GENRE'],s=100,c='blue',label = "horror")
plt.scatter(iv.loc[iv['cluster']==3,'SCORE'],iv.loc[iv['cluster']==3,'MAIN_GENRE'],s=100,c='grey',label = "romance")
plt.scatter(iv.loc[iv['cluster']==4,'SCORE'],iv.loc[iv['cluster']==4,'MAIN_GENRE'],s=100,c='brown',label ="fantasy")

plt.title("Results of K Means Clustering")
plt.xlabel("SCORE")
plt.ylabel("MAIN GENRE")
plt.legend()
plt.show()


# In[ ]:




