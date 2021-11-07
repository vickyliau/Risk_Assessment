import os
import json
import shutil


# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

output_model_path = config["output_model_path"]
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# function for deployment
def store_model_into_pickle(model):
    # copy the latestscore.txt, ingestfiles.txt file into deployment directory
    shutil.copy(output_model_path+"/lastestscore.txt", prod_deployment_path + "/lastestscore.txt")
    shutil.copy(output_model_path+"/ingestedfiles.txt", prod_deployment_path + "/ingestedfiles.txt")
    shutil.copy(output_model_path+"/trainedmodel.pkl", prod_deployment_path + "/trainedmodel.pkl")
