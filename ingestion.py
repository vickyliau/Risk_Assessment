import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob
from itertools import chain


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    datalists1 = glob.glob(input_folder_path+'/*.csv')
    tab=pd.DataFrame()
    for i in range(len(datalists1)):
        tab = tab.append(pd.read_csv(datalists1[i]))
    tab = tab.reset_index(drop=True)
    tab = tab.drop_duplicates()
    tab.to_csv(output_folder_path+'/finaldata.csv',index=False)
    
    datalists2 = glob.glob(test_data_path+'/*.csv')
    tab=pd.DataFrame()
    for i in range(len(datalists2)):
        tab = tab.append(pd.read_csv(datalists2[i]))
    tab = tab.reset_index(drop=True)
    tab = tab.drop_duplicates()
    tab.to_csv(output_folder_path+'/testdata.csv',index=False)


    results = [datalists1,datalists2]
    
    outfile = open(prod_deployment_path+'/ingestedfiles.txt', 'w')
    outfile.writelines(chain(*results))

if __name__ == '__main__':
    merge_multiple_dataframe()

