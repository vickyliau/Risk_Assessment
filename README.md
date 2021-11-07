# A Dynamic Risk Assessment System

This project deploys the API a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. 


## Project Steps Overview

### Data ingestion. 

Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.

### Training, scoring, and deploying. 

Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.

### Diagnostics. 

Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.

### Reporting. 

Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.

### Process Automation. 

Create a script and cron job that automatically run all previous steps at regular intervals.

## API Instructions

### run python app.py in your terminal
### type in http://172.25.98.229:8000/ in your brower
### get the output results by running python apicalls.py


