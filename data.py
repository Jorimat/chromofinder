# Import data from Labelbox + Extract data labelled as 'Done' in Labelbox

import labelbox as lb
import os
import pickle
import urllib
import numpy as np
from itertools import chain
import skimage
from PIL import Image
import matplotlib.pyplot as plt
from functions import *

    
# Import data from Labelbox as JSON file via Python client
LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbG02ODR1cm4wZWhpMDcwcTk1dHdoYWxwIiwib3JnYW5pemF0aW9uSWQiOiJja2g0azIxdjIwdTRzMDcyMTFlcGlvdXcyIiwiYXBpS2V5SWQiOiJjbG02OWpvaGMwaXFuMDd5a2gyNmEybWt1Iiwic2VjcmV0IjoiYzdlNmNhNDU5MTc1ZTE1ODNlZGEzMDQ0ODM0MTI0MmIiLCJpYXQiOjE2OTM5MTU0ODksImV4cCI6MjMyNTA2NzQ4OX0.OOLIh-HLRYhliKp_L3rtwOWprDpsQo8i03lK4PXRLFY'
PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'
client = lb.Client(api_key=LB_API_KEY)
project = client.get_project(PROJECT_ID)
export_params = {
    "attachments": True,
    "metadata_fields": True,
    "data_row_details": True,
    "project_details": True,
    "label_details": True,
    "performance_details": True,
    "interpolated_frames": True
}

export_task = project.export_v2(params=export_params)
export_task.wait_till_done()
if export_task.errors:
    print(export_task.errors)    
Data_ALL = export_task.result

print("The list contains data from",len(Data_ALL),"images.")

# Save data of all images
with open('Data_ALL.pkl', 'wb') as file:
    pickle.dump(Data_ALL, file)

# Directory with all images
os.mkdir("ALL_Images")

for i in range(len(Data_ALL)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    url = Data_ALL[i]["data_row"]["row_data"]
    urllib.request.urlretrieve(url, os.path.join("ALL_Images",external_id)) 

# Collect data of annotated images
Data_ANNOTATED = []

for i in range(len(Data_ALL)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    workflow_status = Data_ALL[i]["projects"][PROJECT_ID]["project_details"]["workflow_status"]
    # Add image if it is labelled as 'DONE' and if it has annotations
    if workflow_status == "DONE":
        # Check whether image has annotations
        try:
            points = len(Data_ALL[i]["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"])
        except IndexError:
            print(external_id,"["+str(i)+"]","has not been annotated")
        else:
            if points != 0:
                Data_ANNOTATED.append(Data_ALL[i])
            else:
                print(external_id,"["+str(i)+"]","has not been annotated")

print("The list contains data from", len(Data_ANNOTATED),"annotated images.")

# Save data of annotated images
with open('Data_ANNOTATED.pkl', 'wb') as file:
    pickle.dump(Data_ANNOTATED, file)   
        
