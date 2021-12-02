#!/usr/bin/env python
# coding: utf-8

# #NUMBER 1: DATA PREPARATION

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print("Done importing libraries")


# In[5]:


#path_to_file= "C:/Users/necbanking/Desktop/2.1 modular/fifa_AI/"
df=pd.read_csv("players_20.csv")
df_19=pd.read_csv("players_19.csv")

df.head()


# In[6]:


df_19.head()


# In[7]:


df = df.drop('dob', axis =1)
df = df.drop('weight_kg', axis =1)
df = df.drop('international_reputation', axis =1)
df = df.drop('real_face', axis =1)
df = df.drop('release_clause_eur', axis =1)
df = df.drop('player_tags', axis =1)
df = df.drop('team_jersey_number', axis =1)
df = df.drop('loaned_from', axis =1)
df = df.drop('joined', axis =1)
df = df.drop('contract_valid_until', axis =1)
df = df.drop('nation_position', axis =1)
df = df.drop('nation_jersey_number', axis =1)
df = df.drop('player_traits', axis =1)
df = df.drop('sofifa_id', axis =1)
df = df.drop('long_name', axis =1)


# In[8]:


df_19 = df_19.drop('dob', axis =1)
df_19 = df_19.drop('weight_kg', axis =1)
df_19 = df_19.drop('international_reputation', axis =1)
df_19 = df_19.drop('real_face', axis =1)
df_19 = df_19.drop('release_clause_eur', axis =1)
df_19 = df_19.drop('player_tags', axis =1)
df_19 = df_19.drop('team_jersey_number', axis =1)
df_19 = df_19.drop('loaned_from', axis =1)
df_19 = df_19.drop('joined', axis =1)
df_19 = df_19.drop('contract_valid_until', axis =1)
df_19 = df_19.drop('nation_position', axis =1)
df_19 = df_19.drop('nation_jersey_number', axis =1)
df_19 = df_19.drop('player_traits', axis =1)
df_19 = df_19.drop('sofifa_id', axis =1)
df_19 = df_19.drop('long_name', axis =1)


# #NUMBER 2: CORRELATION

# In[9]:


#splitting data
train_data, test_data=train_test_split(df,test_size=0.25)
print("Leingth of training data is:"+str(len(train_data)))
print("Leingth of test data is:"+str(len(test_data)))


# In[10]:


#selecting features
target_feature='overall'

#finding features that arecorrelated to the overall column
feature_corr=train_data.corr(method='pearson')[target_feature]
feature_corr=feature_corr.sort_values(ascending=False)
#print thetop ten correlations with the target value
print(feature_corr[1:21])
corr_matrix = df.corr()
corr_matrix['overall'].sort_values(ascending=False)

##


# #NUMBER 3: REGRESSION MODEL
# 

# In[11]:


#Training Rgression model
features=corr_matrix['overall'].sort_values(ascending=False)
features=['potential','value_eur','wage_eur','attacking_short_passing','skill_long_passing','age','skill_ball_control','skill_curve','skill_moves','attacking_volleys']
X_train=df[features]
y_train=df['overall']
r = LinearRegression()
r.fit(X_train,y_train )
print(r.score(X_train,y_train))


# In[12]:


#copying top 20 relavent features to be used by model
features=feature_corr[1:14].index.tolist()
print(features)


# In[13]:


#training the model
x_train=train_data[features]
y_train=train_data[target_feature]

#replace all empty cells with zero
x_train.fillna(0,inplace=True)

#using the LinearRegression method to build the model
model=LinearRegression().fit(x_train,y_train)
#print score
print("Score:"+str(model.score(x_train,y_train)))


# #NUMBER 4: A PROCESS OF OPTIMISATION

# In[14]:


#testing the model usint the 25% of the players_20.csv(df) dataframe
#sort test data first
test_data=test_data.sort_values([target_feature], ascending=False)

x_test=test_data[features]
x_test.fillna(0,inplace=True)
y_test=test_data[target_feature]

#start predicting
y_predict=model.predict(x_test)
#add new column called predicted 
test_data['predicted']=y_predict
rating=((y_predict-y_test)/y_test*100)
#add anew column called accuracy
test_data['difference']=rating
test_data[["short_name","overall","predicted","difference"]]


# In[16]:


#preproccessing features
df_19['potential'] = pd.to_numeric(df_19['potential'],errors='coerce')
df_19['value_eur'] = pd.to_numeric(df_19['value_eur'],errors='coerce')
df_19['wage_eur'] = pd.to_numeric(df_19['wage_eur'],errors='coerce')
df_19['attacking_short_passing'] = pd.to_numeric(df_19['attacking_short_passing'],errors='coerce')
df_19['skill_long_passing'] = pd.to_numeric(df_19['skill_long_passing'],errors='coerce')
df_19['age'] = pd.to_numeric(df_19['age'],errors='coerce')
df_19['skill_ball_control'] = pd.to_numeric(df_19['skill_ball_control'],errors='coerce')
df_19['skill_curve'] = pd.to_numeric(df_19['skill_curve'],errors='coerce')
df_19['skill_moves'] = pd.to_numeric(df_19['skill_moves'],errors='coerce')    
df_19['attacking_volleys'] = pd.to_numeric(df_19['attacking_volleys'],errors='coerce')  


# #NUMBER 5

# In[17]:


#selecting features from the 2019 dataset
features=['potential','value_eur','wage_eur','attacking_short_passing','skill_long_passing','age','skill_ball_control','skill_curve','skill_moves','attacking_volleys']
x_test=df_19[features]
x_test.fillna(0,inplace=True)
y_test=df_19['overall']
predict=r.predict(x_test)
df_19['predicted']=predict
df_19[['short_name','overall','predicted']]


# In[18]:


import pickle


# In[19]:


filename="player_rating.pkl"
outfile=open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# In[ ]:




