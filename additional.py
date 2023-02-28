#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd





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
        
    return df     ### it will return encode dataframe
    
    
# attrdata = pd.read_excel("HR Dummy Dataset.xlsx", sheet_name='Copy of Sheet1')
# order_col = ['job_level','education','employment_status']
# cl_drop = ['sub-department','sexual_orientation', 'hire_date','term_date','term_type','term_reason'] 
# target_variable = 'active_status' 
# encoded = encode_func(attrdata ,order_col, cl_drop , target_variable ) 

#print(attrdata)
# (encoded)
def extension_reader(file_name):                ## if the file is csv then it return 0 else 1
    for i in ['csv']:   
        extension = file_name.split('.')[1]
        if i == extension:
            return 0
        else:
            return 1
