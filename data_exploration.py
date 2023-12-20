# -*- coding: utf-8 -*-
import pickle
from PIL import Image
import requests
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from functions import *
import random

# load data 
with open('Data_ALL.pickle', 'rb') as file:
    Data_ALL = pickle.load(file)
    
with open('Data_LABELABLE.pickle', 'rb') as file:
    Data_LABELABLE = pickle.load(file)  
   
with open('Data_ANNOTATED.pickle', 'rb') as file:
     Data_ANNOTATED = pickle.load(file)    
   

# IMAGE SIZE

# dictionary with name of image as key and image size as value 
dict_sizes = {}

for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    w = Data_ANNOTATED[i]["media_attributes"]["width"]
    h = Data_ANNOTATED[i]["media_attributes"]["height"]
    dict_sizes[external_id] = (w,h)
    
# determine the possible sizes + number of images with specific size
sizes = unique_values(list(dict_sizes.values()))

print("{:^15}{:1}{:^18}".format("SIZE","|", "NUMBER OF IMAGES"))
print("="*35)
for size, n in sizes.items():
    print("{:^15}{:1}{:^18}".format(str(size[0])+" x "+str(size[1]),":", n))


# check image with abnormal size

# find name of image with size 1273 x 947: "Ilex_IL55_IL55 21 11 27apr.jpg"
name = list(dict_sizes)[list(dict_sizes.values()).index((1273,947))]
print(name,"has an abnormal image size.")

# show image
for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    if external_id == name:
        url = Data_ANNOTATED[i]["data_row"]["row_data"]
        r = requests.get(url, stream=True)
        im = Image.open(r.raw)
        im.show()
    
# remove image
remove_image(Data_ANNOTATED,1273,947)
remove_image(Data_LABELABLE,1273,947)   


# AMOUNT OF IMAGES/GENUS

# determine the genus    
# list with names of images
names_labelable = list(Data_LABELABLE[i]["data_row"]["external_id"] for i in range(len(Data_LABELABLE)))
names_annotated = list(Data_ANNOTATED[i]["data_row"]["external_id"] for i in range(len(Data_ANNOTATED)))
# list with genera
genus_labelable = list(genus_finder(names_labelable[i]) for i in range(len(names_labelable)))
genus_annotated = list(genus_finder(names_annotated[i]) for i in range(len(names_annotated)))

# dictionary with amount genus as key and amount as value
genera_labelable = unique_values(genus_labelable)
genera_annotated = unique_values(genus_annotated)
genera = genera_labelable.keys()

# amount of images/genus
print("{:^15}{:1}{:^10}{:^10}".format("GENUS","|", "ALL", "ANNOTATED"))
print("="*36)
for genus in genera:
    n_all = genera_labelable[genus]
    if genus not in genera_annotated.keys():
        n_annotated = 0
    else:
        n_annotated = genera_annotated[genus]
    print("{:15}{:1}{:^10}{:^10}".format(genus,"|",n_all,n_annotated))
    

# NR. ANNOTATED/IMAGE
 
# determine times image is annotated
index = []
print("{:^25}{:1}{:^20}".format("NAME","|", "TIMES ANNOTATED"))
print("{:^25}{:1}{:^10}{:^10}".format("","|", "ALL", "REAL"))
print("="*46)

for i in range(len(Data_ANNOTATED)):
    nr_annotated = len(Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"])
    name = Data_ANNOTATED[i]["data_row"]["external_id"]
    if nr_annotated>1:
        index.append(i)
        real_nr_annotated = nr_annotated
        for annotated in range(nr_annotated):
            skipped = Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"][annotated]["performance_details"]["skipped"]
            if skipped == True:
                real_nr_annotated -= 1
        
        print("{:25}{:1}{:^10}{:^10}".format(name,"|",nr_annotated,real_nr_annotated))
          
# show points for every nr. annotated 
n = random.choice(index)
nt = len(Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"])
w = Data_ANNOTATED[n]["media_attributes"]["width"]
h = Data_ANNOTATED[n]["media_attributes"]["height"]

POINTS = []

for i in range(nt):
    skipped = Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"][i]["performance_details"]["skipped"]
    path_points = Data_ANNOTATED[n]["projects"][PROJECT_ID]["labels"][i]["annotations"]["objects"]
    if skipped == False:
        points = list((path_points[k]["point"]["x"],path_points[k]["point"]["y"]) 
                              for k in (range(len(path_points))))
        POINTS.append(points)

for i in range(len(POINTS)):
    plt.figure()
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.scatter(*zip(*POINTS[i]),s=15, c="r")
    
# only keep annotations that were last updated 
Data_ANNOTATED_real = real_annotations(Data_ANNOTATED, PROJECT_ID)  
            
            
# DISTANCE BETWEEN annotations
# coordinates of annotations
dict_annotations = annotations(Data_ANNOTATED_real, PROJECT_ID)

# smallest possible distance between 2 annotations (for all images)
min_dist_all = 1000

for points in dict_annotations.values():
    min_dist = min(pdist(points))
    if min_dist < min_dist_all:
        min_dist_all = min_dist
        
print("The smallest possible distance between two chromosome centromeres is", min_dist_all)

# distance to nearest annotation for every annotation on all images
dict_distances = {}

for name,points in dict_annotations.items():
    genus = genus_finder(name)
    if genus not in dict_distances:
        dict_distances[genus] = []
    distances = squareform(pdist(points))
    for i in range(np.shape(distances)[0]):
        row = list(distances[i])
        del row[i]
        dict_distances[genus].append(min(row))

all_distances = []
for distances in dict_distances.values():
    for distance in distances:
         all_distances.append(distance)
        
# histogram
plt.figure()
plt.hist(all_distances,200)
plt.title("All genera")
plt.xlabel("Distance to nearest annotation for every annotation")

plt.figure()
for i,genus in enumerate(dict_distances.keys()):
    plt.subplot(3,2,i+1)
    plt.hist(dict_distances[genus],100)
    plt.title(genus)
    plt.xlabel("Distance to nearest annotation for every annotation")  
    plt.tight_layout()

       