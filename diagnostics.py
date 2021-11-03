
import pandas as pd
import numpy as np
import time
import os
import json
from training import train_model
from ingestion import merge_multiple_dataframe

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    testdata = pd.read_csv(dataset_csv_path+'/testdata.csv').iloc[:,1:]
    indep_variable = testdata.copy()
    dep_variable = indep_variable.pop("exited")
    pipe = joblib.load(prod_deployment_path+'/trainedmodel.pkl')
    pred = pipe.predict(indep_variable)
    return pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data = pd.read_csv(dataset_csv_path+'/finaldata.csv').iloc[:,1:]
    df = data.describe()
    df.to_csv(output_folder_path+'/summary.csv',index=False)
    return df #return value should be a list containing all summary statistics

##################Function to calculate the percentage of missing data
def dataframe_missing():
    df = data.isna().sum()/data.shape[0]
    df.to_csv(output_folder_path+'/missing.csv',index=False)
    return df #return value with the percentage of missing data

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time1=time.time()
    merge_multiple_dataframe()
    time2=time.time()
    train_model()
    time3=time.time()
    return [time3-time2, time2-time1] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    # create pip list txt
    os.system('pip list > pip_list.txt')

    # read content into pandas df
    df = pd.read_csv('pip_list.txt', sep=r'\s+', skiprows=[1])
    df['Package']=df['Package'].str.lower()
    df.columns=['Package','version_installed']
    req = pd.read_csv('requirements.txt', sep=r'==', skiprows=[1])
    req.columns=['Package','version_req']
    req['Package']=req['Package'].str.lower()
    mergetab = req.merge(df,how='left',on='Package')
    
    new=[]
    for i in range(mergetab.shape[0]):
        pacs=mergetab['Package'][i]
        os.system('pip index versions '+str(pacs)+' > pip_new.txt')
        df = pd.read_csv('pip_new.txt', sep=r': ')
        new.append(df.iloc[0][0].split(', ')[0])
    mergetab['version_new']=new
    mergetab.to_csv(output_folder_path+'/version.csv',index=False)
    return mergetab

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
