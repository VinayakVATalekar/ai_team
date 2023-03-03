#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from mongo import *
from additional import *

    
def Salary_Conversion(attrdata):
    for j,i in enumerate(attrdata['salary']):
        if i >=40000 and i<= 60000:
            attrdata['salary'][j] = 0
        elif i >60000 and i<= 80000:
            attrdata['salary'][j] = 1
        elif i >80000 and i<= 100000:
             attrdata['salary'][j] = 2
        elif i > 100000 and i<= 120000:
             attrdata['salary'][j] = 3
        elif i > 120000 and i<= 141000:
             attrdata['salary'][j] = 4
    return attrdata


def encode_func(df, order_col,cl_drop,target_variable):
    ### shift target variable to the end
    last_column = df.pop(target_variable)
    df.insert(len(df.columns), target_variable, last_column)
    ### drop unwanted columns
    df = df.drop(columns = cl_drop)
    for col in df.columns:
        
        if df[col].dtypes == 'object':
            
            ## ordinal encoding
            if col in order_col:
                categories = df[col].sort_values().unique()
                mapping = {category: i for i, category in enumerate(categories, start=1)}
                df[col] = df[col].map(mapping)
            ## label encoding
            else:
                df[col] = df[col].astype('category').cat.codes
        
    return df
    
    
def data_generator(df):
    global X_train, X_test,y_train,y_test,x_train_df,x_test_df,x,y
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=4)
    x_train_df = X_train
    x_test_df = X_test
#    X_train, X_test = normalisation_feature(X_train, X_test)
#    return X_train, X_test,y_train,y_test#,x_train_df,x_test_df
    
    
def normalisation_feature(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test) 
    return X_train, X_test
    
    
    

def final_dataframe(x_tr,x_tst,y,df):
    #global pred_prob
    d_f = pd.concat([x_tr,x_tst]).sort_index()
    d_f['active_status'] = y
    df['employee_status_pred'] = d_f['employee_status']
    df['probability_of_leaving'] = d_f['probability_of_leaving']
    return df
    
    
    

    
def main(f_name):
    global pred_prob
    ext=extension_reader(f_name)
    if ext==0:
        attrdata=pd.read_csv(f_name)
    else:
        attrdata=pd.read_excel(f_name)
    # attrdata = pd.read_excel("HR Dummy Dataset.xlsx", sheet_name='Copy of Sheet1')
    oc=finding('order')
    order_col=oc["ord_clm"]
    dc=finding('del')
    cl_drop=dc["del_clm"]
    tar=finding('target')
    target_variable=tar["target"][0]
    # attrdata = pd.read_excel("HR Dummy Dataset.xlsx", sheet_name='Copy of Sheet1')
    # order_col = ['job_level','education','employment_status']
    # cl_drop = ['sub-department','sexual_orientation', 'hire_date','term_date','term_type','term_reason'] 
    # target_variable = 'active_status' 
    encoded = encode_func(attrdata ,order_col, cl_drop , target_variable ) 
    attrdata
    data_generator(encoded)
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train,y_train)
    prediction=dt.predict(X_train)
    y_pred=dt.predict(X_test)
    pred_prob = dt.predict_proba(X_test)
    pred_prob_train = dt.predict_proba(X_train)
    x_train_df['employee_status']=dt.predict(X_train)
    x_test_df['employee_status']=dt.predict(X_test)
    score1=accuracy_score(y_train,prediction)
    score=accuracy_score(y_test,y_pred)
    msg1="training data accuracy is : %f" % (score1)
    msg2="test data accuracy is : %f" % (score)
    
    ## for test data
    prob_test_val = []
    for i in range(len(pred_prob)):
       # print(pred_prob[i][0])
        prob_test_val.append(pred_prob[i][0])
    x_test_df['probability_of_leaving'] = prob_test_val
    
   ## for train data 
    prob_train_val = []
    for k in range(len(pred_prob_train)):
       # print(pred_prob[i][0])
        prob_train_val.append(pred_prob_train[k][0])
    x_train_df['probability_of_leaving'] = prob_train_val
    
    
    print(msg1)
    print(msg2)
    #print(x_train_df)
#     print(pred_prob_train[:10,:])
    Dataframe_Final = final_dataframe(x_train_df,x_test_df,y,attrdata)
    return Dataframe_Final
    


# In[100]:


# df_fnl = main()
# df_fnl


# In[ ]:



 


# In[ ]:





# In[ ]:




