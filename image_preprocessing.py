# -*- coding: utf-8 -*-
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


# DATASET FOR TRAINING, VALIDATION, TESTING (6 genera)
######################################################

# load data 
with open('Data.pkl', 'rb') as file:
    Data = pickle.load(file)
    
    
# make directory
today = str(datetime.now())
today = today[:today.find(" ")]
os.mkdir(today)
today = "2024-03-23"
path = os.path.abspath(today)

# !change names everytime new data is loaded
pickle_data = os.path.join(path,"Data_2.pkl")
pickle_RESIZED_images = os.path.join(path,"Images(2688x2048)_2.pkl")
pickle_RESIZED_NORM_images = os.path.join(path,"Images(2688x2048)_NORM_2.pkl")
pickle_labels = os.path.join(path,"Labels(2688x2048)_2(6).pkl")
pickle_labels_NORM = os.path.join(path,"Labels(2688x2048)_NORM_2(6).pkl") #(A in function Gaussian blobs), NORM: no overlapping Gaussian blob
pickle_coordinates = os.path.join(path,"Coordinates(2688x2048)_2.pkl")
pickle_coordinates_NORM = os.path.join(path,"Coordinates(2688x2048)_NORM_2.pkl")
pickle_genera = os.path.join(path,"Genera_2.pkl")
pickle_genera_NORM = os.path.join(path,"Genera_NORM_2.pkl")

pickle_train_images = os.path.join(path,"TRAIN_Images_2.pkl")
pickle_train_labels = os.path.join(path,"TRAIN_Labels_2.pkl")
pickle_train_coordinates = os.path.join(path,"TRAIN_Coordinates_2.pkl")
pickle_train_genera = os.path.join(path,"TRAIN_Genera_2.pkl")
pickle_test_labels = os.path.join(path,"TEST_Labels_2.pkl")
pickle_test_images = os.path.join(path,"TEST_Images_2.pkl")
pickle_test_coordinates = os.path.join(path,"TEST_Coordinates_2.pkl")
pickle_test_genera = os.path.join(path,"TEST_Genera_2.pkl")


# ANNOTATIONS
# coordinates of points
dict_annotations = annotations(Data, PROJECT_ID, True)

# verify whether coordinates of annotations are correct
names = list(dict_annotations.keys())
name = random.choice(names)
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


# IMAGES AND LABELS

# retrieving all pickle files containing old images
list_pathnames = glob.glob("*\Images*.pkl")

# list with old images
OLD_images = []
for pathname in list_pathnames:
    with open(pathname, 'rb') as file:
        OLD_images.extend(pickle.load(file))
               
data_NEW = []
NEW_images = []
labels = []
coordinates = []
genera = []

for i in range(len(Data)):
    external_id = Data[i]["data_row"]["external_id"]
    genus = genus_finder(external_id)
    
    path = os.path.join("ALL_images",external_id)
    image = cv2.imread(path)
    # resize image
    resized_image = cv2.resize(image,(2688,2048))[:,:,0] 
    
    if len(OLD_images)>0:
    # check if image is new
        old = np.any(np.all(OLD_images == resized_image, axis=(1, 2)))
    else:
        old = False
        
    if len(NEW_images)>0:     
    # check if there is a duplicate
        duplicate = np.any(np.all(NEW_images == resized_image, axis=(1, 2)))
    else:
        duplicate = False
        
    if not old and not duplicate:
        # save data of new image
        data_NEW.append(Data[i])
        # save new, resized image as 2D array (height, width)
        NEW_images.append(resized_image)
        
        # save coordinates of points
        coordinates.append(dict_annotations[external_id])
        
        # save genus
        genera.append(genus)
        
        # make label
        # make figure with same size as image
        shape =  np.shape(resized_image)
        blobs = np.zeros((shape[0], shape[1]))
        sigma_x = sigma_y = 6
        A = 1
        x,y = np.meshgrid(np.arange(0,shape[1],1),np.arange(0,shape[0],1)) 
        # add Gaussian blob to figure for every annotation
        for point in dict_annotations[external_id]:
            x0 = point[0]
            y0 = point[1]
            blobs += gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y)
        labels.append(blobs)
        
print("There are",len(NEW_images),"new images.")
print("There are",len(labels),"labels.")
print("The list contains data from",len(data_NEW),"new images.")
   
# verify whether Gaussian blobs are at the correct position
i = random.choice(range(len(labels)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(NEW_images[i],cmap="gray")
plt.scatter(*zip(*coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(labels[i], cmap="inferno")


# save variables

with open(pickle_data, 'wb') as file:
    pickle.dump(data_NEW, file)
    
with open(pickle_RESIZED_images, 'wb') as file:
    pickle.dump(NEW_images, file) 
    
with open(pickle_labels, 'wb') as file:
    pickle.dump(labels, file)
    
with open(pickle_coordinates, 'wb') as file:
    pickle.dump(coordinates, file)
    
with open(pickle_genera, 'wb') as file:
    pickle.dump(genera, file)
    
# Overlapping Gaussian blobs

# Number of labels with overlapping Gaussian blobs
overlapping = [np.any(labels[i]>1) for i in range(len(labels))]
n = sum(overlapping)
print("There are", str(n),"("+str(round(100*n/len(labels),2))+" %)",
      "labels with overlapping Gaussian blobs.")
indices = [i for i, x in enumerate(overlapping) if x == True]

# Images and labels whithout overlapping Gaussian blobs
for i in sorted(indices, reverse=True):
    print(np.amax(labels[i]))
    #del(NEW_images[i])
    #del(labels[i])
    #del(coordinates[i])
    #del(genera[i])
    del(Data[i])
   
print("There are", str(len(labels)),"labels whithout overlapping Gaussian blobs.")


# save variables
with open(pickle_RESIZED_NORM_images, 'wb') as file:
    pickle.dump(NEW_images, file)   

with open(pickle_labels_NORM, 'wb') as file:
    pickle.dump(labels, file)
    
with open(pickle_coordinates_NORM, 'wb') as file:
    pickle.dump(coordinates, file)
    
with open(pickle_genera_NORM, 'wb') as file:
    pickle.dump(genera, file)
    
       
# NORMALIZATION

# labels/images as 3D array (number of images, height, width)
NEW_images = np.asarray(NEW_images)
NEW_images = NEW_images.astype("float32")/255

labels = np.asarray(labels)
np.amax(labels)
np.amin(labels)
labels = labels.astype("float32")


# check
labels.shape == NEW_images.shape
print(NEW_images.shape)

    
# CHECK FOR RELICATES
# check for replicates in images
for i in range(len(NEW_images)):
    if sum(np.all(NEW_images == NEW_images[i], axis=(1, 2))) > 1:
        print("There is a replicate.")
        
# check for replicates in labels
for i in range(len(labels)):
    if sum(np.all(labels == labels[i], axis=(1, 2))) > 1:
        print("There is a replicate.")


# TEST SET AND TRAIN SET

# shuffle images and labels
perm = np.random.default_rng(seed=42).permutation((len(coordinates)))

shuffled_images = NEW_images[perm]
shuffled_labels = labels[perm]

shuffled_coordinates = []
for i in perm:
    shuffled_coordinates.append(coordinates[i])
    
shuffled_genera = []
for i in perm:
    shuffled_genera.append(genera[i])

# verify whether images and labels are shuffled in the same way
i = random.choice(range(len(shuffled_images)))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(shuffled_images[i],cmap="gray")
plt.scatter(*zip(*shuffled_coordinates[i]),s=15, c="r")
plt.subplot(1,2,2)
plt.imshow(shuffled_labels[i], cmap="inferno")


# split images and labels in train and test set
s = int(0.8*len(shuffled_images))

TRAIN_images = shuffled_images[:s]
TRAIN_labels = shuffled_labels[:s]
TEST_images = shuffled_images[s:]
TEST_labels = shuffled_labels[s:]

len(shuffled_images) == len(TRAIN_images) + len(TEST_images)
len(shuffled_labels) == len(TRAIN_labels) + len(TEST_labels)

TRAIN_coordinates = shuffled_coordinates[:s]
TEST_coordinates = shuffled_coordinates[s:]

TRAIN_genera = shuffled_genera[:s]
TEST_genera = shuffled_genera[s:]

# verify wether the same image is in the train and test set
for i in range(len(TRAIN_images)):
    if sum(np.all(TEST_images == TRAIN_images[i], axis=(1, 2))) > 0:
        print("The same image is in the test set and train set.")

# verify whether labels are with the corect image
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


# images for thesis
index = 1
plt.imsave(os.path.join("Afbeeldingen\Input model","Image.png"), arr=TRAIN_images[index], cmap="gray", format="png")
plt.imsave(os.path.join("Afbeeldingen\Input model","Target.png"), arr=TRAIN_labels[index], format="png")


# save train set and test test as pickle files

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
        
    
# DATASET FOR TESTING ROBUSTNESS (Lavandula)
############################################

# IMAGES
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
        # resize image
        resized_image = cv2.resize(image,(2688,2048))[:,:,0]  
        
        images.append(resized_image)
        chromosome_numbers.append(chromosome_number)
        genotypes.append(genotype)


# NORMALISATION
images = np.asarray(images)
images_norm = []
for image in images:
    images_norm.append(image.astype("float32")/np.amax(image))
images_norm = np.asarray(images_norm)

#os.mkdir("Testset_Lavendel")
os.mkdir(os.path.join("Testset_Lavendel", directory))
path_directory = os.path.join("Testset_Lavendel", directory)

# save as pickle files
with open(os.path.join(path_directory,"TEST_Images_Lavendel.pkl"), 'wb') as file:
    pickle.dump(images_norm, file)
with open(os.path.join(path_directory,"TEST_Ground_thruth_Lavendel.pkl"), 'wb') as file:
    pickle.dump(chromosome_numbers, file)
with open(os.path.join(path_directory,"TEST_Genotypes_Lavendel.pkl"), 'wb') as file:
    pickle.dump(genotypes, file)


##############################################################################
# images for thesis
##############################################################################


# load data 
today = "2024-03-23"
path = os.path.join(os.path.abspath(today),"Data_2.pkl") 
with open(path, 'rb') as file:
    Data = pickle.load(file)
    
dict_annotations = annotations(Data, PROJECT_ID, False)
dict_annotations_resized = annotations(Data, PROJECT_ID, True)


external_id = "Salvia_Snap-1419.jpg"
path = os.path.join("ALL_images",external_id)
image = cv2.imread(path)[:,:]
A = 1

shape =  np.shape(image)
sigma_x = sigma_y = 1.5
x,y = np.meshgrid(np.arange(0,shape[1],1),np.arange(0,shape[0],1)) 
blobs_before = np.zeros((shape[0], shape[1]))
for point in dict_annotations[external_id]:
            x0 = point[0]
            y0 = point[1]
            blobs_before += gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y)
resized_image = cv2.resize(image,(2688,2048))[:,:,0]

shape =  np.shape(resized_image)
sigma_x = sigma_y = 2
x,y = np.meshgrid(np.arange(0,shape[1],1),np.arange(0,shape[0],1)) 
blobs_after = np.zeros((shape[0], shape[1])) 
for point in dict_annotations_resized[external_id]:
            x0 = point[0]
            y0 = point[1]
            blobs_after += gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y)
plots = [image, blobs_before, resized_image, blobs_after]
names = ["Original image", "Original target","Resized image", "Resized target"]
plt.figure()
for i, plot in enumerate(plots):
    name = names[i]
    if name == "Original image" or name == "Resized image":
        cmap = "gray"
    else:
        cmap = "inferno"
    plt.subplot(2,2,i+1)
    plt.imshow(plot, cmap=cmap)
    plt.title(name)
    
plt.figure()
plt.imshow(image[95:130,455:490], cmap='gray')
plt.figure()
plt.imshow(blobs_before[95:130,455:490], cmap='inferno')
plt.figure()
plt.imshow(resized_image[380:510,1770:1900], cmap='gray')
plt.figure()
plt.imshow(blobs_after[380:510,1770:1900], cmap='inferno')

plt.figure()
ax = plt.subplot()
ax.imshow(blobs_after, cmap="inferno", interpolation='none')
rect = patches.Rectangle((1770,380), 130, 130, linewidth=2, edgecolor="red", facecolor="none")
ax.add_patch(rect)
ax.axis("off")
plt.savefig(os.path.join("Afbeeldingen\Input model","Target.pdf"), format="pdf", bbox_inches='tight', pad_inches=0)

plt.imsave(os.path.join("Afbeeldingen\Input model","Gaussian blob original image.png"), arr=blobs_before[95:130,455:490], cmap="inferno", format="png")



with open(pickle_RESIZED_images, 'rb') as file:
    NEW_images = pickle.load(file)
with open(pickle_labels, 'rb') as file:
    labels = pickle.load(file)
    
with open(pickle_RESIZED_NORM_images, 'rb') as file:
    NEW_images = pickle.load(file)
with open(pickle_labels_NORM, 'rb') as file:
    labels = pickle.load(file)
    

with open(pickle_train_images, 'rb') as file:
    TRAIN_images = pickle.load(file)
with open(pickle_train_labels, 'rb') as file:
    TRAIN_labels = pickle.load(file)
with open(pickle_train_coordinates, 'rb') as file:
    TRAIN_coordinates = pickle.load(file)
    
with open(pickle_test_images, 'rb') as file:
    TEST_images = pickle.load(file)
with open(pickle_test_labels, 'rb') as file:
    TEST_labels = pickle.load(file)
with open(pickle_test_coordinates, 'rb') as file:
    TEST_coordinates = pickle.load(file)
with open(pickle_test_genera, 'rb') as file:
    TEST_genera = pickle.load(file)    
