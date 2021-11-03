from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json
import joblib


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = config['prod_deployment_path']

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    testdata = pd.read_csv(dataset_csv_path+'/testdata.csv').iloc[:,1:]
    indep_variable = testdata.copy()
    dep_variable = indep_variable.pop("exited")
    pipe = joblib.load(prod_deployment_path+'/trainedmodel.pkl')
    pred = pipe.predict(indep_variable)
    f1 = f1_score(dep_variable, pred.reshape(pred.shape[0],1))
    #np.savetxt(,np.array(f1))
    with open(prod_deployment_path+'/lastestscore.txt', "w+") as f:
        f.write(str(f1));
if __name__ == '__main__':
    score_model()

