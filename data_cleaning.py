# -*- coding: utf-8 -*-
import numpy as np
import pickle
import skimage
from itertools import chain
from functions import *

PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'

# LOAD DATA
with open('Data_ANNOTATED.pkl', 'rb') as file:
     Data_ANNOTATED = pickle.load(file)    


# REMOVE REPLICATES

# all annotated images (dictionary with name as key and image as value)
dict_images = {}
for i in range(len(Data_ANNOTATED)):
    url = Data_ANNOTATED[i]["data_row"]["row_data"]
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    image = skimage.io.imread(url)[:,:,0]
    dict_images[external_id] = image   

# check for replicates
replicates = []
for name, image in dict_images.items():
    # check if image is already a replicate of another image that was checked before
    if name not in chain(*replicates):
        # number of times that image is present in dataset
        copies = sum(np.all(v == image) for v in dict_images.values())
        if copies > 1:
            # name of replicates
            replicates.append(sorted([k for k, v in dict_images.items() if np.all(v == image)],reverse=True, key=len)) 
            
# dataset without replicates
Data = []            
for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"] 
    if external_id not in chain(*[replicates[i][1:] for i in range(len(replicates))]):
        Data.append(Data_ANNOTATED[i])
        
        
# for each image, only keep annotations that were last reviewed 
Data = real_annotations(Data, PROJECT_ID) 

# REMOVE IMAGE WITH ABNORMAL SIZE (1273x947)
Data = remove_image(Data,1273,947)

# check
dict_sizes = {}
for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    w = Data[i]["media_attributes"]["width"]
    h = Data[i]["media_attributes"]["height"]
    dict_sizes[external_id] = (w,h)
sizes = unique_values(list(dict_sizes.values()))
print("{:^15}{:1}{:^18}".format("SIZE","|", "NUMBER OF IMAGES"))
print("="*35)
for size, n in sizes.items():
    print("{:^15}{:1}{:^18}".format(str(size[0])+" x "+str(size[1]),":", n))
     

# check
for i in range(len(Data)):
    annotated = Data[i]["projects"][PROJECT_ID]["labels"]
    external_id = Data[i]["data_row"]["external_id"]
    nr_annotated = len(annotated)
    if nr_annotated > 1:
        print("Image", external_id, "contains multiple annotations.")
        

# save data
# real annotations without images with abnormal size, replicates
with open("Data.pkl", 'wb') as file:
    pickle.dump(Data, file)
    
    
