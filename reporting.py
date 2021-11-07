import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
import joblib
import glob

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = config["prod_deployment_path"]


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    data = pd.read_csv(dataset_csv_path + "/finaldata.csv").iloc[:, 1:]
    indep_variable = data.copy()
    dep_variable = indep_variable.pop("exited")
    pipe = joblib.load(prod_deployment_path + "/trainedmodel.pkl")
    pred = pipe.predict(indep_variable)
    metrics.ConfusionMatrixDisplay.from_predictions(dep_variable, pred)
    exist = sorted(glob.glob(prod_deployment_path + "/*.png"))[-1]
    if os.path.isfile(exist):
        plt.savefig(
            exist.replace(".png", "")[:-1]
            + str(int(exist.replace(".png", "")[-1]) + 1)
            + ".png"
        )
    else:
        plt.savefig(prod_deployment_path + "/confusionmatrix1.png")


if __name__ == "__main__":
    score_model()
