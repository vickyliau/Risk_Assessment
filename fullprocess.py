
import glob
import pandas as pd
import json
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, dataframe_summary, dataframe_missing, execution_time, outdated_packages_list
from apicalls import apicalls

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']


##################Check and read new data
#first, read ingestedfiles.txt
readfile = list(pd.read_csv(prod_deployment_path+'/ingestedfiles.txt', header=None).T[0])
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
listfile = glob.glob(input_folder_path+'/*.csv')+glob.glob(test_data_path+'/*.csv')

if list(set(readfile) - set(listfile)) != []:
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    print (list(set(readfile) - set(listfile)))
    merge_multiple_dataframe()
    newpred = score_model()['F1 Score']

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    lastpred = float(pd.read_csv(prod_deployment_path+'/lastestscore.txt', header=None).iloc[-1])

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if newpred < lastpred:
        train_model()

        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        
        apicalls()

        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        score_model()
        model_predictions()
        dataframe_summary()
        dataframe_missing()
        execution_time()
        outdated_packages_list()

