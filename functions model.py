import h5py
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import cv2
import random
import skimage.measure as measure
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.exposure as exposure
import sklearn
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# data generator for data augmentation and feeding of images to model
def create_hdf5_generator(
    db_path,
    batch_size,
    augmentations,
    keys = ['features', 'targets'],
    verbose = False
):
    
    """
    Same data augmentation on every instance in a batch.
    Some data augmentation methods (e.g. rotation) must be applied to both features and targets,
    others (e.g. brightness) only to the features.
    """

    db = h5py.File(db_path, 'r')
    db_size = db[keys[0]].shape[0]
    height = db[keys[0]].shape[1]
    width = db[keys[0]].shape[2]
    
    
    while True: 
        for b in np.arange(0, db_size, batch_size):

            if keys == ['features', 'targets']:
                features = db['features'][b:b+batch_size].copy()
                targets = db['targets'][b:b+batch_size].copy()
            
            b_size = len(features)

            # Vertical flip
            if 'random_vertical_flip' in augmentations:
                if np.random.rand() > 0.5:
                    if verbose:
                        print('Vertical flip')
                    features = features[:,::-1,:]
                    targets  = targets[:,::-1,:]
                else:
                    if verbose:
                        print('No vertical flip')

            # Horizontal flip
            if 'random_horizontal_flip' in augmentations:
                if np.random.rand() > 0.5:
                    if verbose:
                        print('Horizontal flip')
                    features = features[:,:,::-1]
                    targets  = targets[:,:,::-1]
                else:
                    if verbose:
                        print('No horizontal flip')
                        
                        
            # Translation
            if 'translation' in augmentations:
                
                translation_vertical   = int(random.uniform(-augmentations['translation'], augmentations['translation']))
                translation_horizontal = int(random.uniform(-augmentations['translation'], augmentations['translation']))
                
                if verbose:
                    print('Vertical translation:', translation_vertical)
                    print('Horizontal translation:', translation_horizontal)  
                    
                features = np.roll(np.roll(features, translation_vertical, axis = 1), translation_horizontal, axis = 2)
                targets = np.roll(np.roll(targets, translation_vertical, axis = 1), translation_horizontal, axis = 2)
                
            # Rotation
            if 'rotation' in augmentations:
                if augmentations['rotation'] > 0:
                    angle = 2*(random.random()-0.5)*augmentations['rotation']
                    if verbose:
                        print('Rotational angle:', angle)
                    M = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
                    features = np.array([cv2.warpAffine(features[i], M, (width, height)) for i in range(b_size)])
                    targets = np.array([cv2.warpAffine(targets[i], M, (width, height)) for i in range(b_size)])
                    # Suboptimal?
                    
            # Brightness
            if 'brightness' in augmentations:
                brightness = 2*(random.random()-0.5)*augmentations['brightness']
                if verbose:
                    print('Brightness:', brightness)
                features += brightness
                if brightness < 0:
                    features = np.maximum(features,0)
                elif brightness > 0:
                    features = np.minimum(features,1)

            # Channel shift
            if 'channelshift' in augmentations:
                for c in range(len(augmentations['channelshift'])):
                    if augmentations['channelshift'][c] != 0:
                        channelshift = 2*(random.random()-0.5)*augmentations['channelshift'][c]
                        if verbose:
                            print('Channel shift:', c, channelshift)
                        features[:,:,c] += channelshift
                        if channelshift < 0:
                            features[:,:,c] = np.maximum(features[:,:,c],0)
                        elif channelshift > 0:
                            features[:,:,c] = np.minimum(features[:,:,c],1)
                            

            yield [features, targets]

            
# make model

# model 0
def initiate_model0(
    h_img,w_img,c_feat,
    num_classes,
    n_contractions,
    dropout_rate,
    size_kernel):
    
    depths = [2**(4+i) for i in range(n_contractions)]
    
    # List to store layers (c) for skip connections
    skip_layers = []
    
    # Input layer
    inputs = tf.keras.layers.Input((h_img, w_img, c_feat))
    
    # Contraction path
    p = inputs
    
    for depth in depths[:-1]:
        c = tf.keras.layers.Conv2D(depth, size_kernel, kernel_initializer='he_normal', padding='same')(p)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        c = tf.keras.layers.Conv2D(depth, size_kernel , kernel_initializer='he_normal', padding='same')(r)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        p = tf.keras.layers.MaxPooling2D((2, 2))(r)
        if dropout_rate!=0:
            p = tf.keras.layers.Dropout(dropout_rate)(p)
        
        skip_layers += [r]
        
    depth = depths[-1]    

    c = tf.keras.layers.Conv2D(depth, size_kernel ,kernel_initializer='he_normal', padding='same')(p)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    c = tf.keras.layers.Conv2D(depth, size_kernel ,kernel_initializer='he_normal', padding='same')(r)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    if dropout_rate!=0:
        r = tf.keras.layers.Dropout(dropout_rate)(r)  
     
    u = r
    
    #Expansive path
    
    for i,depth in enumerate(depths[::-1][1:]):
        u = tf.keras.layers.Conv2DTranspose(depth, (2, 2), strides=(2, 2), padding='same')(u)
        u = tf.keras.layers.concatenate([u, skip_layers[::-1][i]])
        u = tf.keras.layers.BatchNormalization()(u)
        u = tf.keras.layers.ReLU()(u)
        
        # Outputs
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# model 1
def initiate_model1(
    h_img,w_img,c_feat,
    num_classes,
    n_contractions,
    drop_out=False):
    
    depths = [2**(4+i) for i in range(n_contractions)]
    
    # List to store layers (c) for skip connections
    skip_layers = []
    
    # Input layer
    inputs = tf.keras.layers.Input((h_img, w_img, c_feat))
    
    # Contraction path
    p = inputs
    
    for depth in depths[:-1]:
        c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
        if drop_out:
            c = tf.keras.layers.Dropout(0.1)(c)
        c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        p = tf.keras.layers.MaxPooling2D((2, 2))(r)
        
        skip_layers += [c]
        
    depth = depths[-1]    

    c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    if drop_out:
        r = tf.keras.layers.Dropout(0.3)(r)
    c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(r)
    
    #Expansive path
    u = c
    
    for i,depth in enumerate(depths[::-1][1:]):
        u = tf.keras.layers.Conv2DTranspose(depth, (2, 2), strides=(2, 2), padding='same')(u)
        u = tf.keras.layers.concatenate([u, skip_layers[::-1][i]])
        u = tf.keras.layers.BatchNormalization()(u)
        u = tf.keras.layers.ReLU()(u)
    
    # Outputs
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# model 2
def initiate_model2(
    h_img,w_img,c_feat,
    num_classes,
    n_contractions,
    dropout_rate):
    
    depths = [2**(4+i) for i in range(n_contractions)]
    
    # List to store layers (c) for skip connections
    skip_layers = []
    
    # Input layer
    inputs = tf.keras.layers.Input((h_img, w_img, c_feat))
    
    # Contraction path
    p = inputs
    
    for depth in depths[:-1]:
        c = tf.keras.layers.Conv2D(depth, (3, 3), kernel_initializer='he_normal', padding='same')(p)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        c = tf.keras.layers.Conv2D(depth, (3, 3), kernel_initializer='he_normal', padding='same')(r)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        p = tf.keras.layers.MaxPooling2D((2, 2))(r)
        if dropout_rate!=0:
            p = tf.keras.layers.Dropout(dropout_rate)(p)
        
        skip_layers += [r]
        
    depth = depths[-1]    

    c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(p)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(r)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    if dropout_rate!=0:
        r = tf.keras.layers.Dropout(dropout_rate)(r)  
        
    #Expansive path
    
    for i,depth in enumerate(depths[::-1][1:]):
        u = tf.keras.layers.Conv2DTranspose(depth, (2, 2), strides=(2, 2), padding='same')(r)
        u = tf.keras.layers.concatenate([u, skip_layers[::-1][i]])
        c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(u)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(r)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        if dropout_rate!=0:
            r = tf.keras.layers.Dropout(dropout_rate)(r)  
    
        # Outputs
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(r)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# model 3
def initiate_model3(
    h_img,w_img,c_feat,
    num_classes,
    n_contractions,
    dropout_rate):
    
    depths = [2**(4+i) for i in range(n_contractions)]
    
    # List to store layers (c) for skip connections
    skip_layers = []
    
    # Input layer
    inputs = tf.keras.layers.Input((h_img, w_img, c_feat))
    
    # Contraction path
    p = inputs
    
    for depth in depths[:-1]:
        c = tf.keras.layers.Conv2D(depth, (3, 3), kernel_initializer='he_normal', padding='same')(p)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        c = tf.keras.layers.Conv2D(depth, (3, 3), kernel_initializer='he_normal', padding='same')(r)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        d = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(r)
        p = tf.keras.layers.MaxPooling2D((2, 2))(d)

        skip_layers += [r]
        
    depth = depths[-1]    

    c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(p)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(r)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
        
    #Expansive path
    
    for i,depth in enumerate(depths[::-1][1:]):
        u = tf.keras.layers.Conv2DTranspose(depth, (2, 2), strides=(2, 2), padding='same')(r)
        u = tf.keras.layers.concatenate([u, skip_layers[::-1][i]])
        c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(u)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        c = tf.keras.layers.Conv2D(depth, (3, 3),kernel_initializer='he_normal', padding='same')(r)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b) 
    
    # Outputs
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(r)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


# make array binary
def binary(arr, threshold):
    arr_binary = (arr>threshold).astype(int)
    return arr_binary

# normalize pixel values between 0 and 1
def normalization(im):
    max_pv = np.amax(im)
    min_pv = np.amin(im)
    norm_im = (im-min_pv)/(max_pv-min_pv)
    return norm_im

# post-processing
def post_processing(
    im,
    blur_kernel_sz,
    thresh_technique,
    thresh,
    kernel_shape,
    morph_ops,
    order_morph_ops,
    blurring = True):
    
    # blurring
    if blurring:
        kernel = (blur_kernel_sz, blur_kernel_sz)
        blurred = cv2.blur(src=im,ksize=kernel)
    else:
        blurred = im

    # thresholding
    if thresh_technique == cv2.THRESH_BINARY+cv2.THRESH_OTSU:
        T = 255
    else:
        T = thresh
    binary = cv2.threshold(blurred,0,T,thresh_technique)[1]
    
    # morpholoical operations
    for morph_op in order_morph_ops:
        kernel_sz = morph_ops[morph_op]["kernel_size"]
        it = morph_ops[morph_op]["iterations"]
        if kernel_shape == "ellipse":  
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_sz, kernel_sz))
        elif kernel_shape == "square":
            kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
        if morph_op=="erosion":
            binary = cv2.erode(binary, kernel, iterations=it)
        if morph_op=="dilation":
            binary = cv2.dilate(binary, kernel, iterations=it)
        if morph_op=="opening":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel, iterations=it) 
        if morph_op=="closing":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,kernel, iterations=it) 
 
    # labbeling
    labelled = morphology.label(binary)
    
    return labelled

# number of objects
def centroid(im):
    centroids = []
    feats = measure.regionprops(im)
    for feat in feats:
        centroids.append((feat.centroid[1],feat.centroid[0]))
    return centroids


# calculate evaluation metrics

# evaluation of predictions
# calculation of recall
def recall(binary_targets, binary_predictions):
    if binary_targets.shape != binary_predictions.shape:
        print("The targets and predictions don't have the same shape.")
    else:    
        rec = []
        for i in range(len(binary_targets)):
            binary_target = binary_targets[i].flatten()
            binary_prediction = binary_predictions[i].flatten()
            rec.append(sklearn.metrics.recall_score(binary_target, binary_prediction))     
        return rec
            
# calculation of precision 
def precision(binary_targets, binary_predictions):
    if binary_targets.shape != binary_predictions.shape:
        print("The targets and predictions don't have the same shape.")
    else:    
        prec = []
        for i in range(len(binary_targets)):
            binary_target = binary_targets[i].flatten()
            binary_prediction = binary_predictions[i].flatten()
            prec.append(sklearn.metrics.precision_score(binary_target, binary_prediction))
        return prec
            
# calculation of F1-score
def F1(binary_targets, binary_predictions):
    if binary_targets.shape != binary_predictions.shape:
        print("The targets and predictions don't have the same shape.")
    else:
        f1 = []
        for i in range(len(binary_targets)):
            binary_target = binary_targets[i].flatten()
            binary_prediction = binary_predictions[i].flatten()
            f1.append(sklearn.metrics.f1_score(binary_target, binary_prediction))
        return f1
    
# PR-curve
def PR_curve(tagets, predictions, n_thresholds):
    rec_thresh = []
    prec_trhesh = []
    thresholds = np.linspace(0,1,n_thresholds)
    for threshold in thresholds:
        targets_binary = binary(targets)
        predictions_binary = binary(predictions)
        rec_thresh += mean(recall(targets_binary, predictions_binary))
        prec_thresh += mean(precision(targets_binary, predictions_binary))
                     
    return rec_thresh, prec_thresh 

# calculation of Jaccard similiratity coefficient score
def jaccard(binary_targets, binary_predictions):
    if binary_targets.shape != binary_predictions.shape:
        print("The targets and predictions don't have the same shape.")
    else:    
        jaccard = []
        for i in range(len(binary_targets)):
            binary_target = binary_targets[i].flatten()
            binary_prediction = binary_predictions[i].flatten()
            jaccard.append(sklearn.metrics.jaccard_score(binary_target, binary_prediction))
        return jaccard 
    
# evaluation deep learning model
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

# calculate recall for one image
def recall_evaluation(coordinates_real, coordinates_predicted, genus):
    distances_best_matches = find_best_matches(coordinates_real, coordinates_predicted)
    TP = calculate_true_positives(distances_best_matches, genus)
    FN = len(coordinates_real)-TP
    recall_score = TP/(TP+FN)
    return recall_score

# calculate precision for one image
def precision_evaluation(coordinates_real, coordinates_predicted, genus):
    distances_best_matches = find_best_matches(coordinates_real, coordinates_predicted)
    TP = calculate_true_positives(distances_best_matches, genus)
    FP = len(coordinates_predicted)-TP
    precision_score = TP/(TP+FP)
    return precision_score

# calculate f1 for one image
def F1_evaluation(coordinates_real, coordinates_predicted, genus):
    recall_score = recall_evaluation(coordinates_real, coordinates_predicted, genus)
    precision_score = precision_evaluation(coordinates_real, coordinates_predicted, genus)
    if precision_score == 0 or recall_score == 0:
        F1_score = 0
    else:
        F1_score = (2*precision_score*recall_score)/(precision_score+recall_score)
    return F1_score

# calculate recall (modified) for post-processing
def recall_mod(real_pos,pred_pos,crit):
    n = 0
    pred_pos_c = copy.deepcopy(pred_pos)
    for pos in real_pos:
        if len(pred_pos_c)>0:
            distances = cdist(np.array(pred_pos_c).reshape(-1,2),np.array([pos]))
            min_distance = min(distances)
            if min_distance < crit:
                n += 1
                i = list(distances).index(min_distance)
                del pred_pos_c[i]
    rec_mod = n/len(real_pos)
    return rec_mod
                   
# calculate precision (modified) for post-processing                 
def precision_mod(real_pos,pred_pos, crit):
    n = 0
    real_pos_c = copy.deepcopy(real_pos)
    for pos in pred_pos:
        if len(real_pos_c)>0:
            distances = cdist(np.array(real_pos_c).reshape(-1,2),np.array([pos]))
            min_distance = min(distances)
            if min_distance < crit:
                n += 1
                i = list(distances).index(min_distance)
                del real_pos_c[i]
    prec_mod = n/len(pred_pos)
    return prec_mod
 
    
# calculate F1 (modified) for post-processing
def F1_mod(real_pos, pred_pos, crit): 
    rec = recall_mod(real_pos=real_pos, pred_pos=pred_pos, crit=crit)
    prec = precision_mod(real_pos=real_pos, pred_pos=pred_pos, crit=crit)
    
    if prec==0 or rec==0:
        f1_mod = 0
    else:
        f1_mod = (2*prec*rec)/(prec+rec)
            
    return f1_mod

