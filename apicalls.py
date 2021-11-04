import requests
import subprocess
import json

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = config['prod_deployment_path']


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
dd = pd.read_csv(dataset_csv_path+'/testdata.csv').iloc[:,1:]
response1 = requests.post('http://127.0.0.1:8000/prediction',json=dd.to_json()).content #put an API call here
response1 = json.loads(response1)
response2 = requests.get('http://127.0.0.1:8000/scoring').content #put an API call here
response2 = json.loads(response2)
response3 = requests.get('http://127.0.0.1:8000/summarystats').content #put an API call here
response3 = json.loads(response3)
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content #put an API call here
response4 = json.loads(response4)

#combine all API responses
merged_dict = {**response1, **response2, **response3, **response4}
responses = json.dumps(merged_dict) #combine reponses here

#write the responses to your workspace
with open(prod_deployment_path+'/apireturns.txt', 'w') as file:
     file.write(responses)


