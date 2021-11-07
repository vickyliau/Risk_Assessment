from flask import Flask
from scoring import score_model
from diagnostics import (
    loadmodel,
    model_predictions,
    dataframe_summary,
    dataframe_missing,
    execution_time,
    outdated_packages_list,
)
import json
import os
import joblib
import pandas as pd

# Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = config["prod_deployment_path"]


@app.route("/")
def index():
    return "Welcome Risk Assessment API!"


# #Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict(data=pd.read_csv(dataset_csv_path + "/finaldata.csv").iloc[:, 1:]):
    indep_variable = data.copy()
    indep_variable.pop("exited")
    pipe = loadmodel()
    pred = pipe.predict(indep_variable)
    pred = pred.tolist()

    return json.dumps({"prediction": pred})


# #Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    f1 = score_model()
    return f1  # add return value (a single F1 score number)


# #Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    # check means, medians, and modes for each column
    df = dataframe_summary()
    return df  # return a list of all calculated summary statistics


# #Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing and percent NA values
    df = dataframe_missing()
    execution_time()
    outdated_packages_list()
    return df.to_dict()  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
