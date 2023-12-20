# -*- coding: utf-8 -*-
import pickle
import os
from PIL import Image
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import random
import glob
import urllib
import requests


# load data 
with open('Data_ANNOTATED.pickle', 'rb') as file:
    Data_ANNOTATED = pickle.load(file)

PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'

# remove image(s) with abnormal size
remove_image(Data_ANNOTATED,1273,947)


# NEW REVIEWED IMAGES

# name of images which where already annotated
list_pathnames = glob.glob("RESIZED_Images_*\*")
list_names = list(list_pathnames[i][17:] for i in range(len(list_pathnames)))

# !change names everytime new data is loaded
pickle_data_ANNOTATED = "Data_ANNOTATED_1.pickle"
folder_RESIZED_images = "RESIZED_Images_1"
pickle_RESIZED_images = "RESIZED_Images_1.pickle"
pickle_labels = "Labels_1.pickle"

pickle_train_images = "TRAIN_Images_1.pickle"
pickle_train_labels = "TRAIN_Labels_1.pickle"
pickle_test_labels = "TEST_Labels_1.pickle"
pickle_test_images = "TEST_Images_1.pickle"


# create list for data of new annotated images 
Data_ANNOTATED_new = []  
for i in range(len(Data_ANNOTATED)):
    external_id = Data_ANNOTATED[i]["data_row"]["external_id"]
    if external_id not in list_names:
        Data_ANNOTATED_new.append(Data_ANNOTATED[i])
print("The list contains data from",len(Data_ANNOTATED_new),"annotated images.")


# LAST UPDATED ANNOTATIONS
Data_ANNOTATED_new_real = real_annotations(Data_ANNOTATED_new, PROJECT_ID)
   
# save variables
with open(pickle_data_ANNOTATED, 'wb') as file:
    pickle.dump(Data_ANNOTATED_new_real, file)

   
# RESIZE IMAGES

# create new folder for resized images 
os.mkdir(folder_RESIZED_images)
# list with images as 2D array (height, width)
RESIZED_images = []
    
for i in range(len(Data_ANNOTATED_new_real)):
    url = Data_ANNOTATED_new_real[i]["data_row"]["row_data"]
    external_id = Data_ANNOTATED_new_real[i]["data_row"]["external_id"]
    image = Image.open(requests.get(url, stream=True).raw)
    # resize image
    resized_image = image.resize((692,520))
    # save image as 2D array (height, width)
    RESIZED_images.append(np.asarray(resized_image)[:,:,0])
    # save resized image in file
    resized_image.save(os.path.join(folder_RESIZED_images,external_id))
    
print("The file contains",len(os.listdir(folder_RESIZED_images)),"resized images.")
    
    
# NORMALIZATION AND CENTERING
# images as 3D array (number of images, height, width)
RESIZED_images = np.asarray(RESIZED_images)
 
RESIZED_images = RESIZED_images.astype("float32")/255
for i, image in enumerate(RESIZED_images):
    RESIZED_images[i] -= np.mean(image)
    RESIZED_images[i] /= np.std(image)

print("The list contains",len(RESIZED_images),"resized images.")

# save variables
with open(pickle_RESIZED_images, 'wb') as file:
    pickle.dump(RESIZED_images, file)


# LABELS

# coordinates of annotations 
dict_annotations = annotations(Data_ANNOTATED_new_real, PROJECT_ID, True)

# verify whether coordinates of annotations are correct
names = list(dict_annotations.keys())
name = random.choice(names)
points = dict_annotations[name]
image = Image.open(os.path.join(folder_RESIZED_images,name))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(image)
plt.scatter(*zip(*points),s=15, c="r")
plt.suptitle(name,x=0.5,y=0.85, size=15)

# Gaussian_blobs
labels = []
for i in range(len(RESIZED_images)):
    image = RESIZED_images[i]
    # make figure with same size as image
    shape =  np.shape(image)
    blobs = np.zeros((shape[0], shape[1]))
    sigma_x = sigma_y = 3
    A = 3
    x,y = np.meshgrid(np.arange(0,shape[1],1),np.arange(0,shape[0],1)) 
    # add Gaussian blob to figure for every annotation
    for point in list(dict_annotations.values())[i]:
        x0 = point[0]
        y0 = point[1]
        blobs += gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y).transpose()  
    labels.append(blobs)
labels = np.asarray(labels) 

# verify whether Gaussian blobs are at the correct position
i = random.choice(range(len(labels)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(RESIZED_images[i],cmap="gray")
plt.subplot(1,2,2)
plt.imshow(labels[i], cmap="inferno")

# save labels as pickle file
with open(pickle_labels, 'wb') as file:
    pickle.dump(labels, file)


# TEST SET AND TRAIN SET

# shuffle images and labels
perm = np.random.default_rng(seed=42).permutation((len(RESIZED_images)))

shuffled_images = RESIZED_images[perm]
shuffled_labels = labels[perm]

# verify whether images and labels are shuffled in the same way
i = random.choice(range(len(shuffled_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(shuffled_images[i],cmap="gray")
plt.subplot(1,2,2)
plt.imshow(shuffled_labels[i], cmap="inferno")


# split images and labels in train and test set
s = int(0.8*len(shuffled_images))

TRAIN_images = shuffled_images[:s]
TRAIN_labels = shuffled_labels[:s]
TEST_images = shuffled_images[s:]
TEST_labels = shuffled_labels[s:]

# verify whether labels are with the corect image
i = random.choice(range(len(TRAIN_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(TRAIN_images[i],cmap="gray")
plt.subplot(1,2,2)
plt.imshow(TRAIN_labels[i], cmap="inferno")

i = random.choice(range(len(TEST_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(TEST_images[i],cmap="gray")
plt.subplot(1,2,2)
plt.imshow(TEST_labels[i], cmap="inferno")

# save train set and test test as pickle files
with open(pickle_train_images, 'wb') as file:
    pickle.dump(TRAIN_images, file)

with open(pickle_train_labels, 'wb') as file:
    pickle.dump(TRAIN_labels, file)
    
with open(pickle_test_images, 'wb') as file:
    pickle.dump(TEST_images, file)
    
with open(pickle_test_labels, 'wb') as file:
    pickle.dump(TEST_labels, file)
    
    



with open(pickle_RESIZED_images, 'rb') as file:
    RESIZED_images = pickle.load(file)

with open(pickle_labels, 'rb') as file:
    labels = pickle.load(file)
    

with open(pickle_train_images, 'rb') as file:
    TRAIN_images = pickle.load(file)

with open(pickle_train_labels, 'rb') as file:
    TRAIN_labels = pickle.load(file)
    
with open(pickle_test_images, 'rb') as file:
    TEST_images = pickle.load(file)
    
with open(pickle_test_labels, 'rb') as file:
    TEST_labels = pickle.load(file)
    
