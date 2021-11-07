import pandas as pd
import os
from sklearn.metrics import f1_score
import json
import joblib
from diagnostics import loadmodel

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = config["prod_deployment_path"]


# Function for model scoring
def score_model(data = pd.read_csv(dataset_csv_path + "/testdata.csv").iloc[:, 1:]):
    indep_variable = data.copy()
    dep_variable = indep_variable.pop("exited")
    pipe = loadmodel()
    pred = pipe.predict(indep_variable)
    f1 = f1_score(dep_variable, pred.reshape(pred.shape[0], 1))
    with open(prod_deployment_path + "/lastestscore.txt", "w+") as f:
        f.write(str(f1))
    return {"F1 Score": f1}


if __name__ == "__main__":
    score_model()
