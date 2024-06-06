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
import random
from PIL import Image
import copy
from itertools import chain
from functions import *

textwidth = 455.24411
figsize=(set_size(textwidth, fraction=0.49)[0], set_size(textwidth, fraction=0.49)[0]*(2048/2688))

#load data
with open('2024-03-23\\TEST_Images_2.pkl','rb') as file:
    images = pickle.load(file)

with open('2024-03-23\\TEST_Coordinates_2.pkl','rb') as file:
    coordinates = pickle.load(file)  

with open('2024-03-23\\TEST_Genera_2.pkl','rb') as file:
    genera = pickle.load(file)
    

# PREDICTED CHROMOSOME NUMBER VIA SEGMENTATION (OTSU THRESHOLDING) + PREDICTED POSITIONS
processing_techniques = {"erosion":{"footprint":morph.square(9)},
                         "dilation":{"footprint":None},
                         "opening":{"footprint":morph.square(11)},
                         "closing":{"footprint":None},
                         "median":{"footprint":morph.square(9)},
                         "unsharp masking":{"radius":30.0,"amount":1.0}}

order_processing =  ["erosion", "opening"] 

chromosome_number, positions_predicted = baseline(images, coordinates, genera, order_processing, processing_techniques)

# ACTUAL POSITIONS
positions_actual = coordinates

genera_unique = ["Agapanthus", "Geranium", "Ilex", "Persicaria","Salvia", "Thalictrum"]
dict_indices_genera = {}
for genus in genera_unique:
    dict_indices_genera[genus] = [i for i in range(len(chromosome_number["Genus"])) if chromosome_number["Genus"][i] == genus]
    

# COMPARE PREDICTED CHROMOSOME NUMBER WITH ACTUAL CHROMSOME NUMBER

# VISUALLY
i = 50
actual_points = positions_actual[i]
predicted_points = positions_predicted[i]
image = images[i]
processed_image = baseline_processing(image, order_processing, processing_techniques)

plt.figure()
plt.imshow(image, cmap="gray")
plt.scatter(*zip(*actual_points),s=100, c="orange")
plt.axis("off")
plt.legend(["Actual chromosome"], fontsize="25")
plt.figure()
plt.imshow(processed_image, cmap="nipy_spectral")
plt.scatter(*zip(*predicted_points),s=100, c="w", marker="X")
plt.axis("off")
plt.legend(["Predicted chromosome"], fontsize="25")

# SCATTERPLOT
fig, ax = plt.subplots(figsize=(set_size(textwidth, fraction=1)))
sns.scatterplot(data=chromosome_number, 
                x="Actual chromosome number",
                y="Predicted chromosome number",
                hue="Genus",
                hue_order=genera_unique,
                ax=ax)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel("Actual chromosome number", fontsize=12)
ax.set_ylabel("Predicted chromosome number", fontsize=12)
n = np.linspace(0, max(max(chromosome_number["Actual chromosome number"]), max(chromosome_number["Predicted chromosome number"])), 1000)
ax.plot(n, n, 'k-')
plt.show()
plt.savefig("Afbeeldingen\Scatterplot baseline.pdf", format="pdf", bbox_inches='tight')

# CONFUSION MATRIX
cm = metrics.ConfusionMatrixDisplay.from_predictions(
    chromosome_number["Actual chromosome number"],
    chromosome_number["Predicted chromosome number"],
    include_values=False, xticks_rotation='vertical')
ax = cm.ax_
fig = cm.figure_
fig.set_tight_layout(True)
ax.set_xticklabels(labels=cm.display_labels,fontsize=8)
ax.set_yticklabels(labels=cm.display_labels,fontsize=8)

# MEAN ABSOLUTE ERROR
print("{:^48s}  {:^24s}".format("CHROMOSOME NUMBER", "DIFFERENCE"))
print("{:^24s}|{:^24s}".format( "MANUAL", "SEGMENTATION"))
print(116*"=")
for i in range(len(chromosome_number["Genus"])):
    print("{:^24d} {:^24d} {:^24d}".format(
          chromosome_number["Actual chromosome number"][i],chromosome_number["Predicted chromosome number"][i], 
          chromosome_number["Actual chromosome number"][i]-chromosome_number["Predicted chromosome number"][i]))      
print("De mean absolute error is:", 
      metrics.mean_absolute_error(chromosome_number["Actual chromosome number"],chromosome_number["Predicted chromosome number"]))

print("{:15s} {:1s} {:^10s}".format("GENUS","|", "MAE"))
print(25*"=")
for genus in dict_indices_genera.keys():
    indices = dict_indices_genera[genus]
    actual = [chromosome_number["Actual chromosome number"][i] for i in indices]
    predicted = [chromosome_number["Predicted chromosome number"][i] for i in indices]
    MAE = metrics.mean_absolute_error(actual,predicted)
    print("{:15s} {:1s} {:^10.2f}".format(genus,"|", MAE))
    
# OVERESTIMATION/UNDERESTIMATION
actual_numbers = list(chromosome_number["Actual chromosome number"])
predicted_numbers = list(chromosome_number["Predicted chromosome number"])
difference = []
difference_abs = []
for i in range(len(actual_numbers)):
    actual_number = actual_numbers[i]
    predicted_number = predicted_numbers[i]
    difference_abs.append(abs(actual_number-predicted_number))
    difference.append(actual_number-predicted_number)
    
print("Min difference:",np.min(difference_abs))
print("Max difference:",np.max(difference_abs))
print("Mean difference:",np.mean(difference_abs))
print("Median difference:",np.median(difference_abs))

print("Overestimation of the chromosome number:", len([i for i in difference if i < 0]))
print("Underestimation of the chromosome number:", len([i for i in difference if i > 0]))
print("Correct prediction:", len([i for i in difference if i == 0]))

# RECALL - PRECISION - F1
dict_evaluation_metrics = {"recall":{"All":[], "Agapanthus":[], "Geranium":[], "Ilex":[], "Persicaria":[],"Salvia":[], "Thalictrum":[]},
                           "precision":{"All":[],"Agapanthus":[], "Geranium":[], "Ilex":[], "Persicaria":[],"Salvia":[], "Thalictrum":[]},
                           "F1":{"All":[],"Agapanthus":[], "Geranium":[], "Ilex":[], "Persicaria":[],"Salvia":[], "Thalictrum":[]}}
    
for i in range(len(genera)):
    genus = genera[i]
    actual_positions = positions_actual[i]
    predicted_positions = positions_predicted[i]
    
    rec = recall(actual_positions, predicted_positions,genus)
    prec = precision(actual_positions, predicted_positions,genus)
    f1 = F1(actual_positions, predicted_positions,genus)
    
    dict_evaluation_metrics["recall"]["All"].append(rec)
    dict_evaluation_metrics["precision"]["All"].append(prec)
    dict_evaluation_metrics["F1"]["All"].append(f1)
    dict_evaluation_metrics["recall"][genus].append(rec)
    dict_evaluation_metrics["precision"][genus].append(prec)
    dict_evaluation_metrics["F1"][genus].append(f1)

# table of evaluation metrics

print("{:15s} {:1s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s} ".format("GENUS","|", "RECALL","std", "PRECISION","std", "F1","std"))
print(75*"=")
for genus in genera_unique:
    rec = np.mean(dict_evaluation_metrics["recall"][genus])
    std_rec = np.std(dict_evaluation_metrics["recall"][genus])
    prec = np.mean(dict_evaluation_metrics["precision"][genus])
    std_prec = np.std(dict_evaluation_metrics["precision"][genus])
    f1 = np.mean(dict_evaluation_metrics["F1"][genus])
    std_f1 = np.std(dict_evaluation_metrics["F1"][genus])
    print("{:15s} {:1s}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f} ".format(genus,"|", rec, std_rec, prec, std_prec, f1, std_f1))   
print(75*"-")
rec_all = dict_evaluation_metrics["recall"]["All"]
prec_all = dict_evaluation_metrics["precision"]["All"]
f1_all = dict_evaluation_metrics["F1"]["All"]
print("{:15s} {:1s} {:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f} ".format("All","|", np.mean(rec_all), np.std(rec_all), np.mean(prec_all), np.std(prec_all), np.mean(f1_all), np.std(f1_all)))

print("{:20s}{:1s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}".format("","|", "MEAN","std", "MEDIAN","MIN","MAX"))
print(70*"=")
print("{:20s}{:1s}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}".format("precision","|",np.mean(prec_all),np.std(prec_all), np.median(prec_all), np.min(prec_all),np.max(prec_all)))
print("{:20s}{:1s}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}".format("recall","|",np.mean(rec_all), np.std(rec_all), np.median(rec_all),np.min(rec_all),np.max(rec_all)))
print("{:20s}{:1s}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}{:^10.2f}".format("F1-score","|",np.mean(f1_all),np.std(f1_all), np.median(f1_all),np.min(f1_all),np.max(f1_all)))


# boxplot of evaluation metrcis
fig, ax = plt.subplots(figsize=set_size(textwidth))
data = [rec_all, prec_all, f1_all]
labels = ["Recall", "Precision", "F1"]
bp = ax.boxplot(data, patch_artist=True,
                labels=labels,
                boxprops=dict(facecolor="steelblue", color="steelblue"),
                whiskerprops=dict(color="steelblue"),
                capprops=dict(color="steelblue"),
                medianprops=dict(color="orange", linewidth=1.5))
for box in bp['boxes']:
    box.set_facecolor("lightsteelblue")  
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
plt.savefig("Afbeeldingen\Boxplot evaluation baseine.pdf", format="pdf", bbox_inches='tight')


# bad/good predictions
f1_good = [i for i, x in enumerate(f1) if x > 0.9]     
rec_good = [i for i, x in enumerate(rec) if x > 0.9]             
prec_good = [i for i, x in enumerate(prec) if x > 0.9]

f1_bad = [i for i, x in enumerate(f1) if x < 0.5]    
rec_bad = [i for i, x in enumerate(rec) if x < 0.5]               
prec_bad = [i for i, x in enumerate(prec) if x < 0.5]


for index in f1_good:
    image = images[index]
    processed_image = baseline_processing(image, order_processing, processing_techniques)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(processed_image, cmap="nipy_spectral")
    plt.scatter(*zip(*positions_actual[index]),s=7, c="r", marker="x")
    plt.scatter(*zip(*positions_predicted[index]),s=7, c="w", marker="x")
    plt.axis("off")


# IMAGE PRE-PROCESSING OPTIMALISATION/HYPERPARAMETER OPTIMALISATION

# VISUAL OPTIMALISATION
test_indices = [random.randint(0,len(images)-1) for i in range(5)]
for i in test_indices:
    test_image = images[i]
    processed_image = baseline_processing(test_image, order_processing, processing_techniques)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(test_image, cmap="gray")
    plt.scatter(*zip(*positions_actual[i]),s=15, c="g")
    plt.subplot(1,2,2)
    plt.imshow(processed_image, cmap="nipy_spectral")
    plt.scatter(*zip(*positions_actual[i]),s=15, c="g")
    plt.scatter(*zip(*positions_predicted[i]),s=15, c="r")
        
# selection of test images
names = external_id(Data, PROJECT_ID)
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


##############################################################################
# images for thesis
##############################################################################

for i in range(len(images)):
    image = images[i]
    plt.imsave(os.path.join("Afbeeldingen\Test images",str(i)+".png"), arr=image, cmap="gray", format="png")


# pipeline baseline method
image = images[115]

# image after unsharp masking filter 
im_uf = filters.unsharp_mask(image,radius=30.0)
plt.imsave(os.path.join("Afbeeldingen\Baseline","Unsharp masking.png"), arr=im_uf, cmap="gray", format="png")

# image after Otsu thresholding
th_otsu = filters.threshold_otsu(im_uf)
chromosomes = (im_uf > th_otsu)
plt.imsave(os.path.join("Afbeeldingen\Baseline","Otsu thresholding.png"), arr=chromosomes, cmap="gray", format="png")

# image after erosion and opening
chromosomes_e = morph.erosion(chromosomes, footprint=morph.square(9))
chromosomes_oe = morph.opening(chromosomes_e, footprint=morph.square(11)) 
plt.imsave(os.path.join("Afbeeldingen\Baseline","Morphological operations.png"), arr=chromosomes_oe, cmap="gray", format="png")
   
# image after labelling   
chromosomes_labelled = measure.label(chromosomes_oe, background=0)
plt.imsave(os.path.join("Afbeeldingen\Baseline","Labelling.png"), arr=chromosomes_labelled, cmap="nipy_spectral", format="png")

# image with predictions
plt.figure(figsize=(2688,2048))
plt.imshow(np.zeros((2048,2688)), cmap="gray")
plt.scatter(*zip(*positions_predicted[115]),s=300, c="w", marker="X")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join("Afbeeldingen\Baseline","Prediction.png"), format="png", bbox_inches='tight', pad_inches=0)


#otsu thresholding
im = io.imread("Afbeeldingen\Otsu thresholding\image_thresholding_Agapanthus_Snap-55.jpg")[:, :, 0]
th_otsu = filters.threshold_otsu(im)
chromosomes = (im> th_otsu)
n_pixels , intensiteiten = exposure.histogram (im)

plt.figure(figsize=(set_size(textwidth, fraction=0.7)[0],set_size(textwidth, fraction=0.7)[0]))
plt.plot(intensiteiten , n_pixels, color='steelblue' )
plt.axvline(x = th_otsu, color = 'darkorange', label = 'axvline - full height')
plt.ylim(ymin = 0 )
plt.xlabel("Intensity ", size=11)
plt.ylabel("Number of pixels ", size=11)
plt.savefig(os.path.join("Afbeeldingen\Otsu thresholding","Histogram.pdf"), format="pdf", bbox_inches='tight')
plt.imsave(os.path.join("Afbeeldingen\Otsu thresholding","Thresholded image.pdf"),arr=chromosomes, cmap="gray", format="pdf")

plt.figure()
plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
plt.plot(intensiteiten , n_pixels, color='steelblue' )
plt.ylim(ymin = 0 )
plt.xlabel(" Intensity ", size=12)
plt.ylabel(" Number of pixels ", size=12)
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


# bad predictions
bad_predictions = [2,54,77,18]
for index in bad_predictions:
    plt.figure(figsize=(2688,2048))
    image = images[index]
    processed_image = baseline_processing(image, order_processing, processing_techniques)
    plt.imshow(processed_image, cmap="nipy_spectral")
    plt.scatter(*zip(*positions_actual[index]),s=150, c="gray")
    plt.scatter(*zip(*positions_predicted[index]),s=200, c="w", marker="X")
    plt.axis("off")
    plt.tight_layout()
    plt.legend(["True chromosome","Predicted chromosome"], fontsize=25)
    plt.savefig(os.path.join("Afbeeldingen\Bad predictions baseline","Prediction "+str(index)+".pdf"), format="pdf", bbox_inches='tight', pad_inches=0)
    plt.imsave(os.path.join("Afbeeldingen\Bad predictions baseline","Original image "+str(index)+".pdf"), arr=image, cmap="gray", format="pdf")

# real chromosome number = predicted chromosome number   
indices_good = []
for i in range(len(chromosome_number["Genus"])):
    if chromosome_number["Actual chromosome number"][i] == chromosome_number["Predicted chromosome number"][i]:
        indices_good.append(i)

i = indices_good[1]
actual_points = positions_actual[i]
predicted_points = positions_predicted[i]
image = images[i]
processed_image = baseline_processing(image, order_processing, processing_techniques)

figsize=(set_size(textwidth, fraction=0.49)[0], set_size(textwidth, fraction=0.49)[0]*(2048/2688))

plt.figure(figsize=(2048,2688))
plt.imshow(image, cmap="gray")
plt.scatter(*zip(*actual_points),s=100, c="orange")
plt.axis("off")
plt.legend(["True chromosome"], fontsize="25")
plt.savefig(os.path.join("Afbeeldingen\Predictions baseline","Actual positions "+str(i)+".pdf"), format="pdf", bbox_inches='tight', pad_inches=0)
plt.figure(figsize=(2048,2688))
plt.imshow(processed_image, cmap="nipy_spectral")
plt.scatter(*zip(*predicted_points),s=100, c="w", marker="X")
plt.axis("off")
plt.legend(["Predicted chromosome"], fontsize="25")
plt.savefig(os.path.join("Afbeeldingen\Predictions baseline","Predicted positions "+str(i)+".pdf"), format="pdf", bbox_inches='tight', pad_inches=0)
        