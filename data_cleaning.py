# Dataset characterization + Performing data cleaning

import numpy as np
import pickle
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functions import *
import random
import skimage
import cv2
import os
from scipy.spatial.distance import pdist, squareform

PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'

# Load data    
with open('Data_ANNOTATED.pkl', 'rb') as file:
     Data_ANNOTATED = pickle.load(file)    

# Dictionary with external id of annotated image as key and image as value
dict_images = {}

for i in range(len(Data_ANNOTATED)):
    url = Data_ANNOTATED[i]["data_row"]["row_data"]
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    image = skimage.io.imread(url)[:,:,0]
    dict_images[external_id] = image
    
print("The dictionary contains", str(len(dict_images)), "images.")    


# Check if an image has multiple versions of annotations
###
 
# Determine the number of times the image is annotated
#---
index = []
print("{:^25}{:1}{:^20}".format("NAME","|", "TIMES ANNOTATED"))
print("{:^25}{:1}{:^10}{:^10}".format("","|", "ALL", "REAL"))
print("="*46)

for i in range(len(Data_ANNOTATED)):
    nr_annotated = len(Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"])
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    if nr_annotated>1:
        index.append(i)
        real_nr_annotated = nr_annotated
        for annotated in range(nr_annotated):
            skipped = Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"][annotated]["performance_details"]["skipped"]
            # Check if the immage is really annotated
            if skipped:
                real_nr_annotated -= 1
        
        print("{:25}{:1}{:^10}{:^10}".format(external_id,"|",nr_annotated,real_nr_annotated))
          
# Plot each version of the annotations
#---
# Get the coordinates of the annotations for each version of the annotations for a random image
n = random.choice(index)
nt = len(Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"])
w = Data_ANNOTATED[n]["media_attributes"]["width"]
h = Data_ANNOTATED[n]["media_attributes"]["height"]

POINTS = []
for i in range(nt):
    skipped = Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"][i]["performance_details"]["skipped"]
    path_points = Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"][i]["annotations"]["objects"]
    # Check whethter the image is really annotated
    if not skipped:
        # Get coordinates of annotations for one version of annotations
        points = list((path_points[k]["point"]["x"],path_points[k]["point"]["y"]) 
                              for k in (range(len(path_points))))
        POINTS.append(points)

# Plot each version of the annotations
for i in range(len(POINTS)):
    plt.figure()
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.scatter(*zip(*POINTS[i]),s=15, c="r")
    
# Check if a version of annotations of an image is multiple times reviewed
#---
for i in range(len(Data_ANNOTATED)):
    annotated = Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"]
    nr_annotated = len(annotated)
    exteral_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    # Check whether image has multiple versions of the annotations
    if nr_annotated>1:
        for j in range(nr_annotated):
            skipped = annotated[j]["performance_details"]["skipped"]
            # Check whether the image is really annotated
            if not skipped:
                # Check number of times one version of annotations is reviewed
                nr_reviewed = len(annotated[j]["label_details"]["reviews"])
                if nr_reviewed > 1:
                    print(external_id,"is reviewed", str(nr_reviewed), "times.")
                    print(list(annotated[j]["label_details"]["reviews"][n]["reviewed_at"] for n in range(nr_reviewed)))
    
# Keep annotations that were last reviewed 
#---
Data_ANNOTATED = real_annotations(Data_ANNOTATED, PROJECT_ID)  


# Check for replicates
###
counter = 0 
replicates = []
Data = []

for name, image in dict_images.items():
    # Check if image is already a replicate of another image that was checked before
    if name not in chain(*replicates):
        # Number of times that image is present in dataset
        copies = sum(np.all(v == image) for v in dict_images.values())
        if copies > 1:
            # Name of replicates
            replicates.append(sorted([k for k, v in dict_images.items() if np.all(v == image)],reverse=True, key=len)) # Sort based on length => only obtain images that have 1 version of annotations
            print("There are", copies, "replicates:")
            print(*replicates[counter], sep = ", ")
            counter += 1

# Check if the number of objects is different between the replicates
#---

# Dictionary with name as key and number of objects as value
dict_nrannotations = {} 

for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    if external_id in chain(*replicates):
        nr_annotations = len(Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"])
        dict_nrannotations[external_id] = nr_annotations
        
# Number of objects per replicate
for r in replicates:
    l = list((r[i],dict_nrannotations[r[i]]) for i in range(len(r)))
    print(' - '.join([f"{replicate[0]}: {replicate[1]}" for replicate in l]))
            
# Dataset without replicates 
#---
for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"] 
    if external_id not in chain(*[replicates[i][1:] for i in range(len(replicates))]):
        Data.append(Data_ANNOTATED[i])

print("There are",str(counter),"images that are multiple times in the original dataset.")
print("The new dataset contains", str(len(Data)),"images.")
# 39 replicates but one image is 3 times in the dataset => new dataset contain 555 images
 

# Check size of images
###

# Dictionary with name of image as key and image size as value 
dict_sizes = {}

for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    w = Data[i]["media_attributes"]["width"]
    h = Data[i]["media_attributes"]["height"]
    dict_sizes[external_id] = (w,h)
    
# Determine the possible sizes + number of images with specific size
sizes = unique_values(list(dict_sizes.values()))

print("{:^15}{:1}{:^18}".format("SIZE","|", "NUMBER OF IMAGES"))
print("="*35)
for size, n in sizes.items():
    print("{:^15}{:1}{:^18}".format(str(size[0])+" x "+str(size[1]),":", n))

# Check image with abnormal size
#---
# Find name of image with size 1273 x 947: "Ilex_IL55_IL55 21 11 27apr.jpg"
name = list(dict_sizes)[list(dict_sizes.values()).index((1273,947))]
print(name,"has an abnormal image size.")

# Show image
for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    if external_id == name:
        url = Data[i]["data_row"]["row_data"]
        r = requests.get(url, stream=True)
        im = Image.open(r.raw)
        im.show() 
        
# Dataset whithout image with abnormal size
#---
# Remove image with abnormal size
Data = remove_image(Data,1273,947)

  
## Determine the distribution of the images across the genera
###

# Determine the genus 
#---
# List with the external id of the images
names_data = list(Data[i]["data_row"]["external_id"] for i in range(len(Data)))
# List with the genera
genus_data = list(genus_finder(names_data[i]) for i in range(len(names_data)))

# Determine the number of images per genus
#---
# Dictionary with genus as key and amount as value
genera_data = unique_values(genus_data)
genera = genera_data.keys()

# Amount of images/genus
print("{:^15}{:1}{:^10}".format("GENUS","|", "AMOUNT"))
print("="*25)
for genus in genera_data:
    n = genera_data[genus]
    print("{:15}{:1}{:^10}".format(genus,"|",n))
