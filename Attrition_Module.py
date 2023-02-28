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
    X_train, X_test = normalisation_feature(X_train, X_test)
#    return X_train, X_test,y_train,y_test#,x_train_df,x_test_df
    
    
def normalisation_feature(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test) 
    return X_train, X_test
    
    
    
def model_selection():
    global X_train, X_test,y_train,y_test
    lr=LogisticRegression(C = 0.1, random_state = 42, solver = 'liblinear')
    dt=DecisionTreeClassifier()
    rm=RandomForestClassifier(random_state= None)
    gnb=GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=3)
    SVM = svm.SVC(kernel='linear')
    for a,b in zip([lr,dt,knn,SVM,rm,gnb],["Logistic Regression","Decision Tree","KNN","SVM","Random Forest","Naive Bayes"]):
        a.fit(X_train,y_train)
        prediction=a.predict(X_train)
        y_pred=a.predict(X_test)
        score1=accuracy_score(y_train,prediction)
        score=accuracy_score(y_test,y_pred)
    
#     msg1="[%s] training data accuracy is : %f" % (b,score1)
#     msg2="[%s] test data accuracy is : %f" % (b,score)
#     print(msg1)
#     print(msg2)
    model_scores={lr:lr.score(X_test,y_test),
    knn:knn.score(X_test,y_test),
    SVM:SVM.score(X_test,y_test),
    rm:rm.score(X_test,y_test),
    dt:dt.score(X_test,y_test),
    gnb:gnb.score(X_test,y_test)
             }
    best_model = max(zip(model_scores.values(), model_scores.keys()))[1],max(zip(model_scores.values(), model_scores.keys()))[0]
    return best_model[0] 
    
    
def final_dataframe(x_tr,x_tst,y):
    d_f = pd.concat([x_tr,x_tst]).sort_index()
    d_f['active_status'] = y
    return d_f
    
    
    
def Bst_Mdl_Ft(sl_mdl):
    global X_train,y_train,X_test,y_test,x_train_df,x_test_df
    model = sl_mdl.fit(X_train,y_train)    
    x_train_df['employee_status']=model.predict(X_train)
    x_test_df['employee_status']=model.predict(X_test)
    score1=accuracy_score(y_train,x_train_df['employee_status'])
    score=accuracy_score(y_test,x_test_df['employee_status'])
    msg1=" training data accuracy is : %f" % (score1)
    msg2=" test data accuracy is : %f" % (score)
    print(msg1)
    print(msg2)
    
    
    
def main(f_name):
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
    
    
    # order_col = ['job_level','education','employment_status']
    # cl_drop = ['sub-department','sexual_orientation', 'hire_date','term_date','term_type','term_reason'] 
    # target_variable = 'active_status' 
    encoded = encode_func(attrdata ,order_col, cl_drop , target_variable ) 
    attrdata
    data_generator(encoded)
    sel_model = model_selection()
    Bst_Mdl_Ft(sel_model)
    Dataframe_Final = final_dataframe(x_train_df,x_test_df,y)
    print( Dataframe_Final )
    return Dataframe_Final
    
    
