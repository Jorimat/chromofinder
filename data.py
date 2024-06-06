# -*- coding: utf-8 -*-
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

    
# ALL IMAGES (= ALL in Labelbox)
    
# export data (information about image annotations) from Labelbox as JSON via Python client
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

print("The list contains data from",len(Data_ALL), "images.")

# save data of all images
with open('Data_ALL.pkl', 'wb') as file:
    pickle.dump(Data_ALL, file)

# directory with all images
os.mkdir("ALL_Images")

for i in range(len(Data_ALL)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    url = Data_ALL[i]["data_row"]["row_data"]
    urllib.request.urlretrieve(url, os.path.join("ALL_Images",external_id))
    

# LABELABLE IMAGES (= IN REVIEW, DONE in Labelbox)

# collect data of labelable images
Data_LABELABLE = []

for i in range(len(Data_ALL)):
    workflow_status = Data_ALL[i]["projects"][PROJECT_ID]["project_details"]["workflow_status"] 
    # add data of labelable images
    if workflow_status == "IN_REVIEW" or workflow_status == "DONE":
        Data_LABELABLE.append(Data_ALL[i])
        
print("The list contains data from", len(Data_LABELABLE),"images.")

# save data of labelable images
with open('Data_LABELABLE.pkl', 'wb') as file:
    pickle.dump(Data_LABELABLE, file)   


# ANNOTATED IMAGES (= DONE in Labelbox)

# collect data of annotated images
Data_ANNOTATED = []

for i in range(len(Data_ALL)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    workflow_status = Data_ALL[i]["projects"][PROJECT_ID]["project_details"]["workflow_status"]
    # only add images DONE in Labelbox with annotations
    if workflow_status == "DONE":
        # check whether image has annotations
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

# save data of annotated images
with open('Data_ANNOTATED.pkl', 'wb') as file:
    pickle.dump(Data_ANNOTATED, file)   
        

# IMAGES IN REVIEW (= IN REVIEW in Labelbox)

# collect data of images in review
Data_INREVIEW = []

for i in range(len(Data_ALL)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    workflow_status = Data_ALL[i]["projects"][PROJECT_ID]["project_details"]["workflow_status"]
    # only add images IN REVIEW in Labelbox
    if workflow_status == "IN_REVIEW":
        Data_INREVIEW.append(Data_ALL[i])

# check for replicates    
dict_images = {}

# all labelable images (dictionary with name as key and image (2D array) as value)
for i in range(len(Data_LABELABLE)):
    url = Data_LABELABLE[i]["data_row"]["row_data"]
    external_id = Data_LABELABLE[i]["data_row"]["external_id"]
    image = skimage.io.imread(url)[:,:,0]
    dict_images[external_id] = image 
    
print("The dictionary contains", str(len(dict_images)), "images.")   

counter = 0 
replicates = []

for i in range(len(Data_INREVIEW)):
    external_id = Data_INREVIEW[i]["data_row"]["external_id"]
    image = dict_images[external_id]
    if external_id not in chain(*replicates):
        # number of times that image is present in dataset
        copies = sum(np.all(v == image) for v in dict_images.values())
        if copies > 1:
            replicates.append([k for k, v in dict_images.items() if np.all(v == image)])
            print("There are", copies, "replicates:")
            print(*replicates[counter], sep = ", ")
            counter += 1         


##############################################################################
# images for thesis
##############################################################################

with open('Data.pkl', 'rb') as file:
     Data = pickle.load(file)    
dict_annotations = annotations(Data, PROJECT_ID, False)

for i in range(len(Data)):
    external_id = Data_ALL[i]["data_row"]["external_id"]
    url = Data_ALL[i]["data_row"]["row_data"]
    if external_id in example_images:
        

example_images = ["Agapanthus_Snap-205.jpg","Geranium_G9_Snap-395.jpg","Ilex_IL16_IL16 21 11 3mei(30).jpg","Persicaria_P11_Snap-1322.jpg","Salvia_Snap-317.jpg","Thalictrum_Snap-39.jpg"]
for example in example_images:
    points = dict_annotations[example]
    image = Image.open(os.path.join("Afbeeldingen\ALL_images",example))
    plt.figure()
    plt.imshow(image)
    plt.scatter(*zip(*points),s=20, c="orange")
    plt.axis("off")
    plt.savefig(os.path.join("Afbeeldingen\MI original + labelled",example+"_labelled"+".pdf"), format="pdf", bbox_inches='tight', pad_inches=0)