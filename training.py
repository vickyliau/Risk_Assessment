from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import joblib

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = config['prod_deployment_path']

#################Function for training the model
def train_model():
    data = pd.read_csv(dataset_csv_path+'/finaldata.csv').iloc[:,1:]
    indep_variable = data.copy()
    dep_variable = indep_variable.pop("exited")
    #use this logistic regression for training
    pipe = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='multinomial', n_jobs=None, penalty='l2',
                    random_state=0, solver='newton-cg', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    pipe.fit(indep_variable, dep_variable)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    joblib.dump(pipe, prod_deployment_path+'/trainedmodel.pkl')
    return pipe

if __name__ == "__main__":
    pipe = train_model()

