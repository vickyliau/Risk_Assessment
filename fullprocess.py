import glob
import pandas as pd
import json
import os
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from diagnostics import (
    model_predictions,
    dataframe_summary,
    dataframe_missing,
    execution_time,
    outdated_packages_list,
)
from apicalls import apicalls

# Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
test_data_path = config["test_data_path"]
output_model_path = config["output_model_path"]
prod_path = config["prod_deployment_path"]


# Check and read new data
def check_new_files():
    # first, read ingestedfiles.txt
    readfile = list(
        pd.read_csv(prod_path + "/ingestedfiles.txt", header=None).T[0]
    )
    # second, determine whether the source data folder has files
    listfile = glob.glob(input_folder_path + "/*.csv") + glob.glob(
        test_data_path + "/*.csv"
    )

    return bool(list(set(readfile) - set(listfile)) != [])

if check_new_files():
    # Deciding whether to proceed, part 1
    merge_multiple_dataframe()
    data = pd.read_csv(dataset_csv_path + "/finaldata.csv")
    data = data.drop(columns=['corporation'])
    newpred = score_model(data=data)["F1 Score"]

    # Checking for model drift
    lastpred = float(
        pd.read_csv(prod_path + "/lastestscore.txt", header=None).iloc[-1]
    )

    # Deciding whether to proceed, part 2
    if newpred < lastpred:
        train_model()

        # Re-deployment
        # if model drift, re-run the deployment.py script

        apicalls()

        # Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        score_model()
        model_predictions()
        dataframe_summary()
        dataframe_missing()
        execution_time()
        outdated_packages_list()
