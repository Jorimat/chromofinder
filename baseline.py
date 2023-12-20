# -*- coding: utf-8 -*-

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import skimage.io as io
import skimage.morphology as morph
import skimage.measure as measure
import skimage.filters as filters
import skimage.exposure as exposure
import glob
import random
from PIL import Image
from scipy.spatial.distance import cdist
import copy
from functions import *


#load data

PROJECT_ID = 'ckvcagl126z850z6ddufi2cqa'

with open('Data_ANNOTATED.pickle', 'rb') as file:
    Data_ANNOTATED = pickle.load(file)
    
remove_image(Data_ANNOTATED,1273,947)
Data_ANNOTATED_real = real_annotations(Data_ANNOTATED, PROJECT_ID)
    

# PREDICTED CHROMOSOME NUMBER VIA SEGMENTATION (OTSU THRESHOLDING)
dict_chromosomes = {"Name":[], "Genus":[], "Actual chromosome number":[], "Predicted chromosome number":[]}
dict_predicted = {}

for i in range(len(Data_ANNOTATED_real)):
    url = Data_ANNOTATED_real[i]["data_row"]["row_data"]
    external_id = Data_ANNOTATED_real[i]["data_row"]["external_id"]
    n_chromosomes = len(
        Data_ANNOTATED_real[i]["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"])
    genus = genus_finder(external_id)
    
    # calculate chromosome number via segmentation after image preprocessing
    im = io.imread(url)[:, :, 0]
    im_mf_uf = filters.median(filters.unsharp_mask(im))
    th_otsu = filters.threshold_otsu(im_mf_uf)
    chromosomes = (im_mf_uf > th_otsu)
    chromosomes_oe = morph.opening(morph.erosion(chromosomes))
    chromosomes_labelled = measure.label(chromosomes_oe, background=0)
    chromosomes_props = measure.regionprops(chromosomes_labelled)
    n_chromosomes_thresholding = len(chromosomes_props)
    
    #determine position of predicted chromosomes
    predicted_positions = list((rp.centroid[1], rp.centroid[0]) for rp in chromosomes_props)
        
   # add name of picture, genus and chromosome number to dictionary
    dict_chromosomes["Name"].append(external_id)
    dict_chromosomes["Genus"].append(genus)
    dict_chromosomes["Actual chromosome number"].append(n_chromosomes)
    dict_chromosomes["Predicted chromosome number"].append(n_chromosomes_thresholding)
    # add name of image, position of predicted chromosomes to dictionary
    dict_predicted[external_id] = predicted_positions
    

# COMPARE PREDICTED CHROMOSOME NUMBER WITH ACTUAL CHROMSOME NUMBER
dict_actual = annotations(Data_ANNOTATED_real,PROJECT_ID ,False)

# VISUALLY
# Plot predicted positions together with the actual  positions on image
names = list(dict_actual.keys())
name = random.choice(names)
actual_points = dict_actual[name]
predicted_points = dict_predicted[name]

image = Image.open(os.path.join("ALL_images",name))
plt.figure()
plt.imshow(image)
plt.scatter(*zip(*actual_points),s=15, c="g")
plt.scatter(*zip(*predicted_points),s=15, c="r")
plt.suptitle(name,x=0.5,y=0.95, size=15)
plt.legend(["Actual", "Predicted"])

# SCATTERPLOT
fig = plt.figure()
sp = sns.scatterplot(data=dict_chromosomes , 
                     x="Actual chromosome number",
                     y="Predicted chromosome number",
                     hue="Genus")
fig.add_axes(sp)
n = np.linspace(0,max(dict_chromosomes["Actual chromosome number"]+dict_chromosomes["Predicted chromosome number"]),1000)
sp.plot(n,n,'k-')
fig.set_tight_layout(True)

# CONFUSION MATRIX
cm = metrics.ConfusionMatrixDisplay.from_predictions(
    dict_chromosomes["Actual chromosome number"],
    dict_chromosomes["Predicted chromosome number"],
    include_values=False, xticks_rotation='vertical')
ax = cm.ax_
fig = cm.figure_
fig.set_tight_layout(True)
ax.set_xticklabels(labels=cm.display_labels,fontsize=8)
ax.set_yticklabels(labels=cm.display_labels,fontsize=8)

# MEAN ABSOLUTE ERROR
print("{:^40s}  {:^48s}  {:^24s}".format("NAME", "CHROMOSOME NUMBER", "DIFFERENCE"))
print("{:^40s}  {:^24s}|{:^24s}".format( "", "MANUAL", "SEGMENTATION"))
print(116*"=")
for i in range(len(reviewed)):
    print("{:40s}  {:^24d} {:^24d}  {:^24d}".format(dict_chromosomes["Name"][i],
          dict_chromosomes["Actual chromosome number"][i],dict_chromosomes["Predicted chromosome number"][i], 
          dict_chromosomes["Actual chromosome number"][i]-dict_chromosomes["Predicted chromosome number"][i]))      
print("De mean absolute error is:", 
      metrics.mean_absolute_error(dict_chromosomes["Actual chromosome number"],dict_chromosomes["Predicted chromosome number"]))

# RECALL - PRECISION - F1

# RECALL
dict_recall = {}
for name in dict_actual.keys():
    recall = 0
    actual_positions = dict_actual[name]
    predicted_positions = copy.deepcopy(dict_predicted[name])
    for point in actual_positions:
        if len(predicted_positions) > 0:
            distances = cdist(np.array(predicted_positions).reshape(-1,2),np.array([point]))
            min_distance = min(distances)
            if min_distance < 10:
                recall += 1
                i = list(distances).index(min_distance)
                del predicted_positions[i]
    dict_recall[name] = recall/len(actual_positions)
    
# PRECISION
dict_precision = {}
for name in dict_actual.keys():
    precision = 0
    actual_positions = copy.deepcopy(dict_actual[name])
    predicted_positions = dict_predicted[name]
    for point in predicted_positions:
        if len(actual_positions) > 0:
            distances = cdist(np.array(actual_positions).reshape(-1,2),np.array([point]))
            min_distance = min(distances)
            if min_distance < 10:
                precision += 1
                i = list(distances).index(min_distance)
                del actual_positions[i]
    dict_precision[name] = precision/len(predicted_positions)  
    
# F1
dict_F1 = {}
for name in dict_recall.keys():
    recall = dict_recall[name]
    precision = dict_precision[name]
    if precision or recall > 0:
        F1 = (2*precision*recall)/(precision+recall)
        dict_F1[name] = F1
    else:
        dict_F1[name] = 0

# visualization
recall = list(dict_recall.values())
precision = list(dict_precision.values())
F1 = list(dict_F1.values())

print("The mean recall is:", np.mean(recall))
print("The mean precision is:", np.mean(precision))
print("The mean F1-value is:", np.mean(F1))

plt.figure()
plt.boxplot([recall, precision, F1], 
            labels=["Recall", "Precison","F1"])


# IMAGE PRE-PROCESSING OPTIMALISATION/HYPERPARAMETER OPTIMALISATION
        
# selection of test images
names = external_id(Data_ANNOTATED, PROJECT_ID)
test_names = random.choices(names, k=5)
file_images = "ALL_Images"
test_images = list(os.path.join(file_images, test_names[i]) for i in range(len(test_names)))

# plot images
im = []

for i in range(len(test_images)):
    im.append((io.imread(test_images[i]))[:, :, 0])
    plt.figure()
    plt.imshow(im[i], cmap="gray")
    plt.axis("off")

# test thresholding filters
for i in range(len(test_images)):
    filters.try_all_threshold(im[i])

# test filters
for i in range(len(test_images)):
    f = [im[i], filters.median(im[i]), filters.unsharp_mask(im[i]),
         filters.unsharp_mask(filters.median(im[i])),
         filters.median(filters.unsharp_mask(im[i]))]
    names = ["No filter", "Median filter", "Unsharp masking filter",
             "Median + unsharp masking filter", "Unsharp masking + median filter"]
    plt.figure(figsize=(25, 8))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(f[i], cmap="gray")
        plt.axis("off")
        plt.title(names[i])
    for i in range(5):
        plt.subplot(2, 5, 6+i)
        th_otsu = filters.threshold_otsu(f[i])
        chromosomes = (f[i] > th_otsu)
        plt.imshow(chromosomes, cmap="gray")
        plt.axis("off")

# test morphological operations
chromosomes = []

for i in range(len(test_images)):
    im[i] = filters.median(filters.unsharp_mask(im[i]))
    th_otsu = filters.threshold_otsu(im[i])
    chromosomes.append(im[i] > th_otsu)
    operator = [morph.erosion(chromosomes[i]), morph.opening(chromosomes[i]),
                morph.opening(morph.erosion(chromosomes[i])),
                morph.erosion(morph.opening(chromosomes[i])),
                morph.closing(morph.erosion(chromosomes[i])),
                morph.closing(morph.opening(chromosomes[i]))]
    names = ["Erosion", "Opening", "Erosion + opening",
             "Opening + erosion", "Erosion + closing", "Opening + closing"]
    plt.figure(figsize=(12, 12))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(operator[i], cmap ="gray")
        plt.axis("off")
        plt.title(names[i])

# feature extraction
for i in range(len(test_images)):
    chromosomes[i] = morph.opening(morph.erosion(chromosomes[i]))
    chromosomes_label = measure.label(chromosomes[i], background=0)
    plt.figure()
    plt.imshow(chromosomes_label, cmap="nipy_spectral")
    plt.axis("off")
    chromosomes_feats = measure.regionprops(chromosomes_label)
    print(len(chromosomes_feats))


# IMAGES FOR THESIS
    
# plot image
im = io.imread("ALL_Images\Agapanthus_Snap-55.jpg")[:, :, 0]
plt.figure()
plt.imshow(im, cmap="gray")
plt.axis("off")

# image after unsharp masking filter and median filter
im_uf = filters.unsharp_mask(im, 3)
plt.figure()
plt.imshow(im_uf, cmap="gray")
plt.axis("off")
im_uf_mf = filters.median(im_uf)
plt.figure()
plt.imshow(im_uf_mf, cmap="gray")
plt.axis("off")

# image after Otsu thresholding
th_otsu = filters.threshold_otsu(im_uf_mf)
chromosomes = (im_uf_mf > th_otsu)

# histogram with threshold
n_pixels , intensiteiten = exposure . histogram (im_uf_mf)
plt.figure()
plt.imshow(chromosomes, cmap="gray")
plt.axis("off")
plt.figure()
plt. plot ( intensiteiten , n_pixels )
plt. ylim ( ymin = 0 )
plt. xlabel (" Intensity ")
plt. ylabel (" Number of pixels ")
plt. tight_layout ()
plt.axvline(x = th_otsu, color = 'r', label = 'axvline - full height')

# image after erosion and opening
chromosomes_e = morph.erosion(chromosomes)
plt.figure()
plt.imshow(chromosomes_e, cmap="gray")
plt.axis("off")
chromosomes_oe = morph.opening(chromosomes_e) 
plt.figure()
plt.imshow(chromosomes_oe, cmap="gray")
plt.axis("off")    

# image after labelling   
chromosomes_labelled = measure.label(chromosomes_oe, background=0)
plt.figure()
plt.imshow(chromosomes_labelled, cmap="nipy_spectral")
plt.axis("off")    

# Otsu thresholding
im = io.imread("image_thresholding_Agapanthus_Snap-55.jpg")[:, :, 0]
th_otsu = filters.threshold_otsu(im)
chromosomes = (im> th_otsu)
n_pixels , intensiteiten = exposure . histogram (im)

plt.figure()
plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
plt.plot ( intensiteiten , n_pixels, color='steelblue' )
plt.ylim ( ymin = 0 )
plt.xlabel (" Intensity ", size=12)
plt.ylabel (" Number of pixels ", size=12)
plt.title("A", fontsize=12, loc='left')
plt.axvline(x = th_otsu, color = 'darkorange', label = 'axvline - full height')
plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1) 
plt.imshow(im, cmap="gray")
plt.axis("off")
plt.title("B", fontsize=12, loc='left')
plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1) 
plt.imshow(chromosomes, cmap="gray")
plt.axis("off")
plt.title("C", fontsize=12, loc='left')
plt.tight_layout()