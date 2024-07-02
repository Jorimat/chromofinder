# Split dataset in set for training and set for testing

import pickle
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import glob
from datetime import datetime
import cv2
import h5py
from functions import *


PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'
    
# Make a directory
today = str(datetime.now())
today = today[:today.find(" ")]
os.mkdir(today)
path = os.path.abspath(today)

# File names to save data
pickle_train_images = os.path.join(path,"TRAIN_Images_2.pkl")
pickle_train_labels = os.path.join(path,"TRAIN_Labels_2.pkl")
pickle_train_coordinates = os.path.join(path,"TRAIN_Coordinates_2.pkl")
pickle_train_genera = os.path.join(path,"TRAIN_Genera_2.pkl")
pickle_test_labels = os.path.join(path,"TEST_Labels_2.pkl")
pickle_test_images = os.path.join(path,"TEST_Images_2.pkl")
pickle_test_coordinates = os.path.join(path,"TEST_Coordinates_2.pkl")
pickle_test_genera = os.path.join(path,"TEST_Genera_2.pkl")


# Load data 
with open('Data.pkl', 'rb') as file:
    Data = pickle.load(file)

# Coordinates of annotations (after rescaling)
dict_annotations = annotations(Data, PROJECT_ID, True)

# Verify whether the coordinates of the annotations are correct
names = list(dict_annotations.keys())
# Choose a random image
name = random.choice(names)
# Get the coordinates of the annotations for that image
points = dict_annotations[name]
image = Image.open(os.path.join("ALL_images",name))
image = image.resize((2688,2048))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(image)
plt.scatter(*zip(*points),s=15, c="r")
plt.suptitle(name,x=0.5,y=0.85, size=15)


# Resize image and make corresponding label (Gaussian blobs)
###
images = []
labels = []
coordinates = []
genera = []

for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    genus = genus_finder(external_id)
    path = os.path.join("ALL_images",external_id)
    image = cv2.imread(path)
    
    # Resize image
    resized_image = cv2.resize(image,(2688,2048))[:,:,0] 
    
    # Save resized image as 2D array (height, width)
    images.append(resized_image) 
    # Save coordinates of points
    coordinates.append(dict_annotations[external_id])   
    # Save genus
    genera.append(genus)
        
    # Make label
    # Make figure with same size as image
    shape =  np.shape(resized_image)
    blobs = np.zeros((shape[0], shape[1]))
    sigma_x = sigma_y = 6
    A = 1
    x,y = np.meshgrid(np.arange(0,shape[1],1),np.arange(0,shape[0],1)) 
    # Add Gaussian blob to figure for every annotation
    for point in dict_annotations[external_id]:
        x0 = point[0]
        y0 = point[1]
        blobs += gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y)
    labels.append(blobs)
        
print("There are",len(images),"images.")
print("There are",len(labels),"labels.")
   
# Verify whether Gaussian blobs are at the correct position 
i = random.choice(range(len(labels)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(images[i],cmap="gray")
plt.scatter(*zip(*coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(labels[i], cmap="inferno")


# Check whether the Gaussian blobs overlap each other 
###

# Check if max pixel value > 1
overlapping = [np.any(labels[i]>1) for i in range(len(labels))]
# Percentage of labels with a max pixel value > 1
n = sum(overlapping)
print("There are", str(n),"("+str(round(100*n/len(labels),2))+" %)",
      "labels with overlapping Gaussian blobs.")
# Indices of labels where max pixel value > 1
indices = [i for i, x in enumerate(overlapping) if x == True]

# Images and labels whithout overlapping Gaussian blobs
for i in sorted(indices, reverse=True):
    print(np.amax(labels[i]))
    del(images[i])
    del(labels[i])
    del(coordinates[i])
    del(genera[i])
    del(Data[i])
   
print("There are", str(len(labels)),"labels whithout overlapping Gaussian blobs.")
    
       
# Normalisation of images and labels
###

# Labels/images as 3D array (number of images, height, width)
NEW_images = np.asarray(NEW_images)
NEW_images = NEW_images.astype("float32")/255

labels = np.asarray(labels)
labels = labels.astype("float32")

# Check
labels.shape == NEW_images.shape
print(NEW_images.shape)


# Split data in test set and train set
###

# Shuffle images and labels
#---
perm = np.random.default_rng(seed=42).permutation((len(labels)))

shuffled_images = images[perm]
shuffled_labels = labels[perm]
shuffled_coordinates = []
for i in perm:
    shuffled_coordinates.append(coordinates[i])  
shuffled_genera = []
for i in perm:
    shuffled_genera.append(genera[i])

# Verify whether images and labels are shuffled in the same way
i = random.choice(range(len(shuffled_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(shuffled_images[i],cmap="gray")
plt.scatter(*zip(*shuffled_coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(shuffled_labels[i], cmap="inferno")


# Split images and labels in train and test set
#---
s = int(0.8*len(shuffled_images))

TRAIN_images = shuffled_images[:s]
TRAIN_labels = shuffled_labels[:s]
TEST_images = shuffled_images[s:]
TEST_labels = shuffled_labels[s:]
TRAIN_coordinates = shuffled_coordinates[:s]
TEST_coordinates = shuffled_coordinates[s:]
TRAIN_genera = shuffled_genera[:s]
TEST_genera = shuffled_genera[s:]

# Check wether the same image is in the train and test set
for i in range(len(TRAIN_images)):
    if sum(np.all(TEST_images == TRAIN_images[i], axis=(1, 2))) > 0:
        print("The same image is in the test set and train set.")

# Verify whether the labels are with the corect image
i = random.choice(range(len(TRAIN_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(TRAIN_images[i],cmap="gray")
plt.scatter(*zip(*TRAIN_coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(TRAIN_labels[i], cmap="inferno")

i = random.choice(range(len(TEST_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(TEST_images[i],cmap="gray")
plt.scatter(*zip(*TEST_coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(TEST_labels[i], cmap="inferno")


# Save train set and test test as pickle files

with open(pickle_train_images, 'wb') as file:
    pickle.dump(TRAIN_images, file)
with open(pickle_train_labels, 'wb') as file:
    pickle.dump(TRAIN_labels, file)
with open(pickle_train_coordinates, 'wb') as file:
    pickle.dump(TRAIN_coordinates, file)
with open(pickle_train_genera, 'wb') as file:
    pickle.dump(TRAIN_genera, file)
    
with open(pickle_test_images, 'wb') as file:
    pickle.dump(TEST_images, file)    
with open(pickle_test_labels, 'wb') as file:
    pickle.dump(TEST_labels, file)
with open(pickle_test_coordinates, 'wb') as file:
    pickle.dump(TEST_coordinates, file)   
with open(pickle_test_genera, 'wb') as file:
    pickle.dump(TEST_genera, file) 
        
    
# Dataset of Lavandula (to test the robustness)
############################################

# Resize the images
###
images = []
chromosome_numbers = []
genotypes = []

directory = "Lavendel_good"
for subdir in os.listdir(directory):
    path_subdir = os.path.join(directory, subdir)
    chromosome_number, genotype = chromosome_nr_Lavandula(subdir)
    for im in os.listdir(path_subdir):
        path_im = os.path.join(path_subdir, im)
        image = cv2.imread(path_im)
        # Resize image
        resized_image = cv2.resize(image,(2688,2048))[:,:,0]  
        
        images.append(resized_image)
        chromosome_numbers.append(chromosome_number)
        genotypes.append(genotype)


# Normalisation of the images and the labels
###
images = np.asarray(images)
images_norm = []
for image in images:
    images_norm.append(image.astype("float32")/np.amax(image))
images_norm = np.asarray(images_norm)

os.mkdir(os.path.join("Testset_Lavendel", directory))
path_directory = os.path.join("Testset_Lavendel", directory)

# Save as pickle files
with open(os.path.join(path_directory,"TEST_Images_Lavendel.pkl"), 'wb') as file:
    pickle.dump(images_norm, file)
with open(os.path.join(path_directory,"TEST_Ground_thruth_Lavendel.pkl"), 'wb') as file:
    pickle.dump(chromosome_numbers, file)
with open(os.path.join(path_directory,"TEST_Genotypes_Lavendel.pkl"), 'wb') as file:
    pickle.dump(genotypes, file)  
