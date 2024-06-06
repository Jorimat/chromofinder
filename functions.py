# -*- coding: utf-8 -*-
import numpy as np
import copy
import skimage.morphology as morph
import skimage.measure as measure
import skimage.filters as filters
from itertools import repeat
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import skimage.io as io
import cv2
import os
from itertools import chain

# round up numbers
def round_up(number):
    if round(number) < number:
        new_number = round(number)+1
    else:
        new_number = round(number)
    return new_number

# determine unique values in list + amount
def unique_values(lijst):
    # dictionary with genus as key and amount as value
    unique = {}
    for value in lijst:
        # check if value is already in dictionary
        if value not in unique.keys():
            # number of times value appears in list
            n = lijst.count(value)
            unique[value] = n
    return unique

# find genus of plant from which chromosomes originate
def genus_finder(external_id):
    if "IL" in external_id:
        genus = "Ilex"
    else:
        genus = external_id[:external_id.find("_")]
    return(genus)

# remove image(s) with abnormal size
def remove_image(dt, width, height):
    dt_copy = copy.deepcopy(dt)
    for i in range(len(dt)):
        w = dt[i]["media_attributes"]["width"]
        h = dt[i]["media_attributes"]["height"]
        if w == width and h == height:
            dt_copy.remove(dt[i])
    return(dt_copy)
            
# only keep annotations that were last reviewed
def real_annotations(annotated_data, PROJECT_ID):
    annotated_data_real = copy.deepcopy(annotated_data)
    for i in range(len(annotated_data)):
        annotated = annotated_data[i]["projects"][PROJECT_ID]["labels"]
        nr_annotated = len(annotated)
        name = annotated_data[i]["data_row"]["external_id"]
        # check whether image is multiple times annotated 
        if nr_annotated>1:
            print(name, "has", nr_annotated, "different annotated images.")
            removed = []
            dates = {}
            for j in range(nr_annotated):
                skipped = annotated[j]["performance_details"]["skipped"]
                nr_objects = len(annotated[j]["annotations"]["objects"])
                # check if annotated image is really labelled
                if skipped:
                    removed.append(j)
                    print("Annotated image",j,"of image", name,"is skipped and contains", nr_objects, "objects.")
            # find last reviewed annotations
                else:
                    nr_reviewed = len(annotated[j]["label_details"]["reviews"])
                    dates_reviewed = []
                    for n in range(nr_reviewed):
                        # find out the date
                        reviewed_at = annotated[j]["label_details"]["reviews"][n]["reviewed_at"]
                        date = reviewed_at[:reviewed_at.find("T")]+" "+reviewed_at[reviewed_at.find("T")+1:reviewed_at.find(".")]
                        dates_reviewed.append(date)
                    # sort the dates and only keep last reviewed
                    dates_reviewed_sorted = sorted(dates_reviewed)
                    date_last_reviewed = dates_reviewed_sorted[-1]
                    dates[date_last_reviewed] = j
                    if nr_reviewed > 1:
                        print("Annotated image",j,"of image", name,"is reviewed",nr_reviewed, "times.")
                        print(dates_reviewed_sorted)
                        print(date_last_reviewed)
                        
            # sort annotations by date
            dates_sorted = dict(sorted(dates.items())) 
            print(dates_sorted) 
            # make list of annotations that are not the newest
            removed.extend(list(dates_sorted.values())[:-1]) 
            # remove annotations that are not the newest
            removed.sort(reverse=True) 
            print(removed)            
            for index in removed:
                del(annotated_data_real[i]["projects"][PROJECT_ID]["labels"][index])
    return(annotated_data_real)

def real_annotations2(annotated_data, PROJECT_ID):
    annotated_data_real = copy.deepcopy(annotated_data)
    for i in range(len(annotated_data)):
        annotated = annotated_data[i]["projects"][PROJECT_ID]["labels"]
        nr_annotated = len(annotated)
        name = annotated_data[i]["data_row"]["external_id"]
        # check whether image is multiple times annotated 
        if nr_annotated>1:
            print(name, "has", nr_annotated, "different annotated images.")
            removed = []
            dates = {}
            for j in range(nr_annotated):
                skipped = annotated[j]["performance_details"]["skipped"]
                nr_objects = len(annotated[j]["annotations"]["objects"])
                # check if annotated image is really labelled
                if skipped:
                    removed.append(j)
                    print("Annotated image",j,"of image", name,"is skipped and contains", nr_objects, "objects.")
            # find last reviewed annotations
                else:
                    reviewed_at = annotated[j]["label_details"]["reviews"][-1]["reviewed_at"]
                    date = reviewed_at[:reviewed_at.find("T")]+" "+reviewed_at[reviewed_at.find("T")+1:reviewed_at.find(".")]
                    dates[date] = j
            # sort annotations by date
            dates_sorted = dict(sorted(dates.items())) 
            print(dates_sorted) 
            # make list of annotations that are not the newest
            removed.extend(list(dates_sorted.values())[:-1]) 
            # remove annotations that are not the newest
            removed.sort(reverse=True) 
            print(removed)            
            for index in removed:
                del(annotated_data_real[i]["projects"][PROJECT_ID]["labels"][index])
    return(annotated_data_real)

# dictionary with name and coordinates of annotations
def annotations(annotated_data, PROJECT_ID, rescale_annotations):
    dict_annotations = {}
    if rescale_annotations:
        width = int(input("Rescale to width:"))
        height = int(input("Rescale to height:"))
    for i in range(len(annotated_data)):
        name = annotated_data[i]["data_row"]["external_id"]
        w = annotated_data[i]["media_attributes"]["width"]
        h = annotated_data[i]["media_attributes"]["height"]
        # save coordinates of annotations
        path_points = annotated_data[i]["projects"][PROJECT_ID]["labels"][0]["annotations"]["objects"]
        if rescale_annotations:
            # resize coordinates 
            points = list((((path_points[k]["point"]["x"])*(width/w)),((path_points[k]["point"]["y"]))*(height/h)) 
                          for k in (range(len(path_points))))  
        else:
            points = list((path_points[k]["point"]["x"],path_points[k]["point"]["y"]) 
                          for k in (range(len(path_points))))
        dict_annotations[name] = points 
    return dict_annotations

# list with names of images
def external_id(annotated_data, PROJECT_ID):
    external_ids = []
    for i in range(len(annotated_data)):
        external_id = annotated_data[i]["data_row"]["external_id"]
        external_ids.append(external_id)
    return external_ids


# chromosome number of Lavundula
def chromosome_nr_Lavandula(name):
    if name == "Lavendel_L4":
        chromosome_nr = 44
        genotype = "L. dentata 'Ploughmen's blue'"
    elif name == "Lavendel_L3":
        chromosome_nr = 44 
        genotype = "L. dentata var. candicans"
    elif name == "Lavendel_L27":
        chromosome_nr = 28 
        genotype = "L. stoechas 'Kew Red"
    elif name == "Lavendel_L104":
        chromosome_nr = 48  
        genotype = "L. stoechas 'Van Gogh's babies'"
    elif name == "Lavendel_L108":
        chromosome_nr = 50 
        genotype = "L. lanata"               
    elif name == "Lavendel_L126":
        chromosome_nr =  22
        genotype = "L. multifida"
        
    return(chromosome_nr, genotype)
        
# Gaussian blob
def gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y):
    '''Calculate the values of an unrotated Gauss function given positions
    in x and y in a mesh grid'''
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2) -(y-y0)**2/(2*sigma_y**2))

# pre-processing
def preprocessing(image):
    image = filters.median(filters.unsharp_mask(image))
    threshold = filters.threshold_otsu(image)
    objects = (image>threshold)
    objects_labelled = measure.label(objects, background=0)
    return objects_labelled

# identify single chromosomes
def identification_chromosomes(objects_labelled, coordinates, 
                       min_distance, min_area, max_area,max_area_bbox, max_area_diff):
    
    single_chromosomes = []
    
    objects_props = measure.regionprops(objects_labelled)
    for label in objects_props:
        area = label.area
        area_bbox = label.area_bbox
        centroid = tuple(list(label.centroid)[::-1])
        distances = cdist(coordinates, list(tuple(repeat(centroid,len(coordinates)))))
        area_diff = area_bbox-area

        if np.min(distances)<min_distance and area<max_area and area>min_area and area_bbox<max_area_bbox and area_diff<max_area_diff:
            single_chromosomes.append(label)
    
    return single_chromosomes

# optimal paramaters for extraction of single chromosomes
def optimal_parameters(genus):
    if genus == "Ilex":
        min_distance = 50*2
        min_area = 200*4
        max_area = 600*4
        max_area_bbox = 800*4
        max_area_diff = 250*4
    
    if genus == "Agapanthus":
        min_distance = 100*2
        min_area = 2000*4
        max_area = 7000*4
        max_area_bbox = 8000*4
        max_area_diff = 3000*4
        
    if genus == "Geranium":
        min_distance = 50*2
        min_area = 200*4
        max_area = 2000*4
        max_area_bbox = 2200*4
        max_area_diff = 400*4
        
    if genus == "Persicaria":
        min_distance = 50*2
        min_area = 200*4
        max_area = 2800*4
        max_area_bbox = 3000*4
        max_area_diff = 500*4
    
    if genus == "Salvia":
        min_distance = 50*2
        min_area = 1000*4
        max_area = 3500*4
        max_area_bbox = 4000*4
        max_area_diff = 800*4

    if genus == "Thalictrum":
        min_distance = 50*2
        min_area = 500*4
        max_area = 2000*4
        max_area_bbox = 2200*4
        max_area_diff = 400*4
        
    return min_distance, min_area, max_area, max_area_bbox, max_area_diff

# baseline
def baseline(list_images, list_coordinates_real, list_genera, order_processing, processing_techniques , resize=False):
    dict_chromosome_number = {"Genus":[], "Actual chromosome number":[], "Predicted chromosome number":[]}
    list_coordinates_predicted = []
    
    if resize:
        width = int(input("Rescale to width:"))
        height = int(input("Rescale to height:"))
        
    
    for i in range(len(list_images)):
        genus = list_genera[i]
        im = list_images[i]

        if resize:
            # resize image
            im = cv2.resize(im,(width, height))
            
        # determine real chromsome number
        n_chromosomes_real = len(list_coordinates_real[i])  
        
        # calculate chromosome number via segmentation after image preprocessing
        chromosomes = baseline_processing(im, order_processing, processing_techniques)
        chromosomes_feats = measure.regionprops(chromosomes)
        n_chromosomes_predicted = len(chromosomes_feats)
        
        #determine position of predicted chromosomes
        predicted_positions = list((rp.centroid[1], rp.centroid[0]) for rp in chromosomes_feats)
            
       # add name of picture, genus and chromosome number to dictionary
        dict_chromosome_number["Genus"].append(genus)
        dict_chromosome_number["Actual chromosome number"].append(n_chromosomes_real)
        dict_chromosome_number["Predicted chromosome number"].append(n_chromosomes_predicted)
        # add name of image, position of predicted chromosomes to dictionary
        list_coordinates_predicted.append(predicted_positions)
        
    return dict_chromosome_number, list_coordinates_predicted

def baseline_processing(im, order_processing, processing_techniques):
    radius = processing_techniques["unsharp masking"]["radius"]
    amount = processing_techniques["unsharp masking"]["amount"]
    im = filters.unsharp_mask(im, radius=radius, amount=amount)
    #footprint = processing_techniques["median"]["footprint"]
    #im = filters.median(im,footprint=footprint)
    th_otsu = filters.threshold_otsu(im)
    im = (im > th_otsu)
    for technique in order_processing:
        footprint = processing_techniques[technique]["footprint"]
        if technique == "erosion":
            im = morph.binary_erosion(im,footprint=footprint)
        elif technique == "dilation":
            im = morph.binary_dilation(im,footprint=footprint)
        elif technique == "opening":
            im = morph.binary_opening(im,footprint=footprint)
        elif technique == "closing":
            im = morph.binary_closing(im,footprint=footprint)
    chromosomes = measure.label(im, background=0)

    return chromosomes

# criteria (= median width) to determine if two points are close enough together for the determination of the evaluation metrics
def criteria_evaluation_metric(genus):
    if genus == "Agapanthus":
        criteria = 88
    elif genus == "Geranium":
        criteria = 39
    elif genus == "Persicaria":
        criteria = 44
    elif genus == "Salvia":
        criteria = 53
    elif genus == "Thalictrum":
        criteria = 50
    elif genus == "Ilex":
        criteria = 41
    else:
        print("The critria for the given genus is not known") 
    return criteria
        
# find best match between two sets of points by minimising the sum of the distances (Hungarian algorithm)
def find_best_matches(points1, points2):
    
    # calculate distance matrix
    distance_matrix = cdist(points1, points2)
    
    # solve the linear sum assignment problem: find the correct matches by minimizing the sum of the distances
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # find the coordinates corresponding with the correct matches
    #best_matches = [(points1[i],points2[j]) for i, j in zip(row_ind, col_ind)]
    distances_best_matches = [distance_matrix[i,j] for i, j in zip(row_ind, col_ind)]
    
    return distances_best_matches

# calculate true positives (used in calculation of evalution metrics)
def calculate_true_positives(distances_best_matches, genus):
    criteria = criteria_evaluation_metric(genus)
    true_positives = sum([distances_best_matches[i] < criteria for i in range(len(distances_best_matches))])
    return true_positives

# evaluation metrics

# calculate recall for one image
def recall(coordinates_real, coordinates_predicted, genus):
    distances_best_matches = find_best_matches(coordinates_real, coordinates_predicted)
    TP = calculate_true_positives(distances_best_matches, genus)
    FN = len(coordinates_real)-TP
    recall_score = TP/(TP+FN)
    return recall_score

# calculate precision for one image
def precision(coordinates_real, coordinates_predicted, genus):
    distances_best_matches = find_best_matches(coordinates_real, coordinates_predicted)
    TP = calculate_true_positives(distances_best_matches, genus)
    FP = len(coordinates_predicted)-TP
    precision_score = TP/(TP+FP)
    return precision_score

# calculate f1 for one image
def F1(coordinates_real, coordinates_predicted, genus):
    recall_score = recall(coordinates_real, coordinates_predicted, genus)
    precision_score = precision(coordinates_real, coordinates_predicted, genus)
    if precision_score == 0 or recall_score == 0:
        F1_score = 0
    else:
        F1_score = (2*precision_score*recall_score)/(precision_score+recall_score)
    return F1_score


# extract all values from nested dictionary
def values_dict(d):
    return list(chain.from_iterable(
        [values_dict(v) if isinstance(v, dict) else [v]
         if isinstance(v, str) else v for v in d.values()]))



def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim