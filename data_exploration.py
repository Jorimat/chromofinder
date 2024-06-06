# -*- coding: utf-8 -*-
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

# load data 
with open('Data_ALL.pkl', 'rb') as file:
    Data_ALL = pickle.load(file)
    
with open('Data_LABELABLE.pkl', 'rb') as file:
    Data_LABELABLE = pickle.load(file)  
   
with open('Data_ANNOTATED.pkl', 'rb') as file:
     Data_ANNOTATED = pickle.load(file)    

with open('2024-03-23\Data_2.pkl', 'rb') as file:
     Data = pickle.load(file)      

# IMAGES
# all annotated images (dictionary with name as key and image as value)
dict_images = {}

for i in range(len(Data_ANNOTATED)):
    url = Data_ANNOTATED[i]["data_row"]["row_data"]
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    image = skimage.io.imread(url)[:,:,0]
    dict_images[external_id] = image
    
print("The dictionary contains", str(len(dict_images)), "images.")    
  

# REPLICATES
counter = 0 
replicates = []
Data = []

# check for replicates
for name, image in dict_images.items():
    # check if image is already a replicate of another image that was checked before
    if name not in chain(*replicates):
        # number of times that image is present in dataset
        copies = sum(np.all(v == image) for v in dict_images.values())
        if copies > 1:
            # name of replicates
            replicates.append(sorted([k for k, v in dict_images.items() if np.all(v == image)],reverse=True, key=len)) # sort based on length => only obtain images that have 1 annotation
            print("There are", copies, "replicates:")
            print(*replicates[counter], sep = ", ")
            counter += 1

# check if the number of objects is different between the replicates
Data_ANNOTATED = real_annotations(Data_ANNOTATED, PROJECT_ID) 

# number of objects (dictionary with name as key and number of objects as value)
dict_nrannotations = {} 

for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    if external_id in chain(*replicates):
        nr_annotations = len(Data_ANNOTATED[i]["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"])
        dict_nrannotations[external_id] = nr_annotations
        
# number of objects per replicate
for r in replicates:
    l = list((r[i],dict_nrannotations[r[i]]) for i in range(len(r)))
    print(' - '.join([f"{replicate[0]}: {replicate[1]}" for replicate in l]))
            
# dataset without replicates            
for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"] 
    if external_id not in chain(*[replicates[i][1:] for i in range(len(replicates))]):
        Data.append(Data_ANNOTATED[i])

print("There are",str(counter),"images that are multiple times in the original dataset.")
print("The new dataset contains", str(len(Data)),"images.")
# 39 replicates but one image is 3 times in the dataset => new dataset contain 555 images
 

# IMAGE SIZE

# dictionary with name of image as key and image size as value 
dict_sizes = {}

for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    w = Data[i]["media_attributes"]["width"]
    h = Data[i]["media_attributes"]["height"]
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
for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    if external_id == name:
        url = Data[i]["data_row"]["row_data"]
        r = requests.get(url, stream=True)
        im = Image.open(r.raw)
        im.show() 
# remove image
Data = remove_image(Data,1273,947)
Data_ANNOTATED = remove_image(Data_ANNOTATED,1273,947)
Data_labelable = remove_image(Data_LABELABLE,1273,947) 

  
# AMOUNT OF IMAGES/GENUS

# determine the genus    
# list with names of images
names_data = list(Data[i]["data_row"]["external_id"] for i in range(len(Data)))
names_labelable = list(Data_LABELABLE[i]["data_row"]["external_id"] for i in range(len(Data_LABELABLE)))
names_annotated = list(Data_ANNOTATED[i]["data_row"]["external_id"] for i in range(len(Data_ANNOTATED)))
# list with genera
genus_data = list(genus_finder(names_data[i]) for i in range(len(names_data)))
genus_labelable = list(genus_finder(names_labelable[i]) for i in range(len(names_labelable)))
genus_annotated = list(genus_finder(names_annotated[i]) for i in range(len(names_annotated)))

# dictionary with genus as key and amount as value
genera_data = unique_values(genus_data)
genera_labelable = unique_values(genus_labelable)
genera_annotated = unique_values(genus_annotated)
genera = genera_labelable.keys()

# amount of images/genus
print("{:^15}{:1}{:^10}".format("GENUS","|", "AMOUNT"))
print("="*25)
for genus in genera_data:
    n = genera_data[genus]
    print("{:15}{:1}{:^10}".format(genus,"|",n))
    
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

for i in range(len(Data)):
    nr_annotated = len(Data[i]["projects"][PROJECT_ID]["labels"])
    external_id = Data[i]["data_row"]["external_id"]
    if nr_annotated>1:
        index.append(i)
        real_nr_annotated = nr_annotated
        for annotated in range(nr_annotated):
            skipped = Data[i]["projects"][PROJECT_ID]["labels"][annotated]["performance_details"]["skipped"]
            # check if annotated immage is really labelled
            if skipped:
                real_nr_annotated -= 1
        
        print("{:25}{:1}{:^10}{:^10}".format(external_id,"|",nr_annotated,real_nr_annotated))
          
# show points for every nr. annotated 
n = random.choice(index)
nt = len(Data[n]["projects"][PROJECT_ID]["labels"])
w = Data[n]["media_attributes"]["width"]
h = Data[n]["media_attributes"]["height"]

POINTS = []

for i in range(nt):
    skipped = Data[n]["projects"][PROJECT_ID]["labels"][i]["performance_details"]["skipped"]
    path_points = Data[n]["projects"][PROJECT_ID]["labels"][i]["annotations"]["objects"]
    if not skipped:
        points = list((path_points[k]["point"]["x"],path_points[k]["point"]["y"]) 
                              for k in (range(len(path_points))))
        POINTS.append(points)

for i in range(len(POINTS)):
    plt.figure()
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.scatter(*zip(*POINTS[i]),s=15, c="r")
    
# check if labelled image is multiple times reviewed   
for i in range(len(Data)):
    annotated = Data[i]["projects"][PROJECT_ID]["labels"]
    nr_annotated = len(annotated)
    exteral_id = Data[i]["data_row"]["external_id"]
    # check whether image is multiple times annotated 
    if nr_annotated>1:
        for j in range(nr_annotated):
            skipped = annotated[j]["performance_details"]["skipped"]
            # check if labelled image is multiple times reviewed
            if not skipped: 
                nr_reviewed = len(annotated[j]["label_details"]["reviews"])
                if nr_reviewed > 1:
                    print(external_id,"is reviewed", str(nr_reviewed), "times.")
                    print(list(annotated[j]["label_details"]["reviews"][n]["reviewed_at"] for n in range(nr_reviewed)))
    
# only keep annotations that were last reviewed 
Data = real_annotations(Data, PROJECT_ID)  
            
            
# DISTANCE BETWEEN points
# coordinates of points
dict_annotations = annotations(Data, PROJECT_ID, True)

# smallest possible distance between 2 points
min_dist_all = 1000

for points in dict_annotations.values():
    min_dist = min(pdist(points))
    if min_dist < min_dist_all:
        min_dist_all = min_dist
        
print("The smallest possible distance between two chromosome centromeres is", min_dist_all)

# distance to nearest point for every point on all images
dict_distances = {}

for name,points in dict_annotations.items():
    genus = genus_finder(name)
    if genus not in dict_distances:
        dict_distances[genus] = []
    # calculate the distance between two points for all points
    distances = squareform(pdist(points))
    for i in range(np.shape(distances)[0]): # np.shape(distances)[0] = number of points
        row = list(distances[i])
        # row[i]=0 because it is the distance between point xi and xi
        del row[i]
        dict_distances[genus].append(min(row))

all_distances = []
for distances in dict_distances.values():
    for distance in distances:
         all_distances.append(distance)
        
# histogram
plt.figure(figsize=set_size(textwidth))
n, bins, patches = plt.hist(all_distances,int(max(all_distances)), color="steelblue")
patches[37].set_fc('orange')
plt.xlim(0,200)
#plt.title("All genera")
plt.xlabel("Distance to nearest chromosome (pixels)")
plt.show()
plt.savefig(os.path.join("Afbeeldingen\Histogram distances.pdf"), format="pdf", bbox_inches='tight', pad_inches=0)

plt.figure()
for i,genus in enumerate(dict_distances.keys()):
    plt.subplot(3,2,i+1)
    plt.hist(dict_distances[genus],100)
    plt.title(genus)
    plt.xlabel("Distance to nearest point for every point")  
    plt.tight_layout()


# labels
with open("2024-02-27\Images_NORM_1.pkl", 'rb') as file:
    images = pickle.load(file)   
with open("2024-02-27\Labels_NORM_1(1.5).pkl", 'rb') as file:
    labels = pickle.load(file)
    

# als threshold voor recall en precision pixelwaarde nemen van Gaussian blob op een afstand van 5 pixels van het centrum
textwidth = 455.24411
blobs = np.zeros((30,30))
sigma_x = sigma_y = 6
A = 1
x,y = np.meshgrid(np.arange(0,41,1),np.arange(0,41,1)) 
x0 = 20
y0 = 20
blob = gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y)
plt.figure(figsize=set_size(textwidth))
plt.imshow(blob, cmap="inferno")
plt.xlim(0,40)
plt.ylim(40,0)
plt.xticks([0,5,10,15,20,25,30,35,40])
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig(os.path.join("Afbeeldingen\Gaussian blob.pdf"), format="pdf", bbox_inches='tight', pad_inches=0)
    
p0_p = np.mean([np.sum(labels[i]>0.0001)/np.sum(labels[i]==0) for i in range(len(labels))])*100
p0_ptotal = np.mean([np.sum(labels[i]>0.0001)/(512*672) for i in range(len(labels))])*100
print("The ratio of pixels with a value > 0.0001 to pixels with a value = 0 is:",np.round(p0_p,2))
print(np.round(p0_ptotal,2), "percent of the pixels has a value bigger than 0.0001.")



# WIDTH OF CHROMOSOMES +-30

# open pickle files 
with open("2024-03-23\Data_2.pkl", 'rb') as file:
    Data = pickle.load(file)

dict_annotations = annotations(Data, PROJECT_ID, True)

dict_images_genera = {}

for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    genus = genus_finder(external_id)
    path = os.path.join("ALL_images",external_id)
    image = cv2.imread(path)
    resized_image = cv2.resize(image,(2688,2048))[:,:,0] 
    points = dict_annotations[external_id]
    if genus not in dict_images_genera.keys():
        dict_images_genera[genus] = {external_id:{ "image":resized_image, "points": points}}
    else:
        dict_images_genera[genus].update({external_id:{ "image":resized_image, "points": points}})
        
# finding optimal parameters for identification single chromosomes/genera  
    
genus = list(dict_images_genera.keys())[5] 
names = list(dict_images_genera[genus].keys())
names = random.sample(names,10)

for name in names:
    image = dict_images_genera[genus][name]["image"]
    points = dict_images_genera[genus][name]["points"]
    
    # pre-processing
    chromosomes_labelled = preprocessing(image)
    
    # identification of single chromosomes
    single_chromosomes = identification_chromosomes(chromosomes_labelled, points, 
                               100,2000,8000,88000,1600)
    #plot
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.scatter(*zip(*points),s=15, c="r")
    plt.subplot(1,2,2)
    plt.imshow(chromosomes_labelled, cmap="nipy_spectral")
    for chromosome in single_chromosomes:
        min_row, min_col, max_row, max_col = chromosome.bbox
        plt.gca().add_patch(Rectangle((min_col,min_row),max_col-min_col,max_row-min_row,linewidth=1,edgecolor='r',facecolor='none'))
    

# determination of width of single chromosomes
dict_width = {}

genera = list(dict_images_genera.keys()) 

for genus in genera:
    names = list(dict_images_genera[genus].keys()) 
    for name in names:
        image = dict_images_genera[genus][name]["image"]
        points = dict_images_genera[genus][name]["points"]
        
        # pre-processing
        chromosomes_labelled = preprocessing(image)
        
        # identification of single chromosomes
        min_distance, min_area, max_area, max_area_bbox, max_area_diff = optimal_parameters(genus)
        single_chromosomes = identification_chromosomes(chromosomes_labelled, points, 
                                   min_distance,min_area,max_area,max_area_bbox,max_area_diff)
        
        # width
        for single in single_chromosomes:
            if genus not in dict_width.keys():
                dict_width[genus] = []
            dict_width[genus].append(single.axis_minor_length)   

# width
print("{:^15}{:1}{:^32}".format("GENUS","|", "WIDTH"))
print("{:15}{:1}{:^8}{:^8}{:^8}{:^8}".format("","|", "MIN", "MAX", "MEAN", "MEDIAN"))
print("="*48)
for genus, widths in dict_width.items():
    mini = round(min(widths))
    maxi = round(max(widths))
    mean = round(np.mean(widths))
    median = round(np.median(widths))
    print("{:15}{:1}{:^8}{:^8}{:^8}{:^8}".format(genus, "|", mini, maxi, mean, median))
    
# percentage chromosomes with width smaller than criterion
criterion = 39

print("{:15}{:1}{:^8}".format("","|", "%"))
print("="*24)
for genus, widths in dict_width.items():
    per = np.round(sum(np.asarray(widths)<criterion)/len(widths),2)
    print("{:15}{:1}{:^8}".format(genus, "|",per))


# NUMBER OF CHROMOSOMES PER IMAGE
with open("2024-03-23\Data_2.pkl", 'rb') as file:
    Data = pickle.load(file)

dict_annotations = annotations(Data, PROJECT_ID, False)

dict_chromosomes = {"All":[]}

for external_id in dict_annotations.keys():
    genus = genus_finder(external_id)
    nr_chromosomes = len(dict_annotations[external_id])
    dict_chromosomes["All"].append(nr_chromosomes)
    if genus not in dict_chromosomes.keys():
        dict_chromosomes[genus] = [nr_chromosomes]
    else:
        dict_chromosomes[genus].append(nr_chromosomes)
        
print("{:20s}{:1s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}".format("genus","|", "MEAN", "std", "MEDIAN","MIN","MAX"))
print(70*"=")
for genus in dict_chromosomes.keys():
    chromosomes = dict_chromosomes[genus]
    print("{:20s}{:1s}{:^10.0f}{:^10.0f}{:^10.0f}{:^10.0f}{:^10.0f}".format(genus,"|",np.mean(chromosomes),np.std(chromosomes), np.median(chromosomes), np.min(chromosomes),np.max(chromosomes)))
    if genus == "All":
        print(70*"-")
