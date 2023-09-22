import glob, os, shutil, re, random
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import scipy.ndimage as snd
import tensorflow as tf
import tensorflow.keras.layers as layers
import keras




def extractXml(fileXml):
    '''
    Extract info from an xml file, the silly way.
    '''
    
    tags = ['xmin', 'xmax', 'ymin', 'ymax', 'name']
    res = {}
    for tag in tags:
        res[tag] = []
    res['n_obj'] = 0

    with open(fileXml) as file_:
        lines = file_.readlines()
        for line in lines:
            for tag in tags:
                if f"<{tag}>" in line:
                    value = line.split(f"<{tag}>")[1].split(f"</{tag}>")[0]
                    # print(tag, value, line)
                    res[tag] += [value]
                    if tag == 'name':
                        res['n_obj'] += 1

    return res


def extractTxt(fileTxt, w, h):

    with open(fileTxt) as file_:
        lines = file_.readlines()
            
    res = []

    for line in lines:
    
        values = [float(x) for x in line.strip('\n').split(' ')]

        hmin = int((values[1]*h - values[3]*h/2))
        vmin = int((values[2]*w - values[4]*w/2))
        hmax = int((values[1]*h + values[3]*h/2))
        vmax = int((values[2]*w + values[4]*w/2))
        label_num = str(int(values[0]))
        
        res += [{
            'xmin' : hmin,
            'ymin' : vmin,
            'xmax' : hmax, 
            'ymax' : vmax,
            'label_num' : label_num
        }]

    return res


def gaussian_blob(x, y, A, x0, y0, sigma_x, sigma_y):
    '''Calculate the values of an unrotated Gauss function given positions
    in x and y in a mesh grid'''
    return A*np.exp(-(x-x0)**2/(2*sigma_x**2) -(y-y0)**2/(2*sigma_y**2)).transpose()



def make_data_one_file_central(
    files,
    w_out,
    h_out,
    n_pos_crops_per_image,
    n_neg_crops_per_image,
    max_tries = 1000,
    plot_everything = False
):

    # Load images
    imgBefore = plt.imread(files['fileJpgBefore'])
    img       = plt.imread(files['fileJpg'])
    imgAfter  = plt.imread(files['fileJpgAfter']) 
    
    # Make full size features
    diff = 2*img - imgBefore - imgAfter
    features_full = np.concatenate([img, diff], axis = 2)

    # Load labels
    labels = extractTxt(files['fileTxt'], img.shape[0], img.shape[1])  
    
    # Make full size target
    target_full = np.zeros(img.shape[:2])

    for label in labels:

        xmin = label['xmin']
        xmax = label['xmax']
        ymin = label['ymin']
        ymax = label['ymax']

        xmid = int((xmin+xmax)/2)
        ymid = int((ymin+ymax)/2)

        x = np.linspace(0,img.shape[0]-1,img.shape[0])
        y = np.linspace(0,img.shape[1]-1,img.shape[1])

        x0 = xmid
        y0 = ymid

        radius_blob = ((xmax-xmin)+(ymax-ymin))/2    
        sig = radius_blob/2

        A = 1
        Xg, Yg = np.meshgrid(x, y)
        target_full += gaussian_blob(Xg, Yg, A, y0, x0, sig, sig)
        
    target_full /= target_full.max()
    
    if plot_everything:
        plt.figure(figsize=(15,10))
        plt.subplot(131)
        plt.imshow(features_full[:,:,:3])
        plt.subplot(132)
        plt.imshow(features_full[:,:,3:])
        plt.subplot(133)
        plt.imshow(target_full)
        plt.show()
    
    
    # Make random crops of features and target
    pos_feats = []
    neg_feats = []
    pos_trgs  = []
    neg_trgs  = []
    
    counter = 0
    
    while (len(pos_feats) < n_pos_crops_per_image or len(neg_feats) < n_neg_crops_per_image) and counter < max_tries:
        
        counter += 1

        x_min_crop = np.random.randint(0, features_full.shape[0]-w_out)
        y_min_crop = np.random.randint(0, features_full.shape[1]-h_out)

        features_crop = features_full[x_min_crop:x_min_crop+w_out, y_min_crop:y_min_crop+h_out, :]
        target_crop   =   target_full[x_min_crop:x_min_crop+w_out, y_min_crop:y_min_crop+h_out]
            
        if target_crop.max() == 1:
            if len(pos_feats) < n_pos_crops_per_image:
                pos_feats += [features_crop]
                pos_trgs  += [target_crop]
                
        else:
            if len(neg_feats) < n_neg_crops_per_image:
                target_crop = np.zeros(target_crop.shape) 
                neg_feats += [features_crop]
                neg_trgs  += [target_crop]
                
                
    # Append
    features_array = np.array(pos_feats + neg_feats)/255
    target_array   = np.array(pos_trgs + neg_trgs)
    
    if plot_everything:
        for i in range(features_array.shape[0]):
            plt.figure(figsize=(15,10))
            plt.subplot(131)
            plt.imshow(features_array[i,:,:,:3])
            plt.subplot(132)
            plt.imshow(features_array[i,:,:,3:])
            plt.subplot(133)
            plt.imshow(target_array[i,:,:])
            plt.show()
    
    return features_array, target_array



def make_data_central(
    files,
    w_out, 
    h_out,
    n_pos_crops_per_image,
    n_neg_crops_per_image,
    fileHDF,
    plot_everything = False
):
    
    c_feat = 6
    n_per_file = (n_pos_crops_per_image+n_neg_crops_per_image)
    n_instances = len(files)*n_per_file

    # Start HDF5 file
    db = h5py.File(fileHDF, 'w')
    features = db.create_dataset('features', (n_instances, w_out, h_out, c_feat), dtype='float32') 
    targets  = db.create_dataset('targets',  (n_instances, w_out, h_out), dtype='float32') 

    # Make features and targets for all files
    for f in range(len(files)):
        print(f, '/', len(files))
        # print(files[f])
        features_array, target_array = make_data_one_file_central(
            files[f],
            w_out,
            h_out,
            n_pos_crops_per_image,
            n_neg_crops_per_image,
            max_tries = 1000,
            plot_everything = plot_everything
        )

        # Store in HDF5 file
        # print('features_array:', features_array.shape)
        n_inst = features_array.shape[0]
        features[f*n_per_file:f*n_per_file+n_inst, :, :, :]  = features_array
        targets [f*n_per_file:f*n_per_file+n_inst, :, :]     = target_array
        
        # WARNING: Some features and targets full of zeroes may remain

    # Close HDF5 file
    db.close()


    
    
# WARNING: This generator loops forever; set a steps_per_epoch in model.fit

# def create_hdf5_generator(db_path, batch_size):
#     db = h5py.File(db_path, 'r')
#     db_size = db['features'].shape[0]
#     while True: # loop through the dataset indefinitely
#         for i in np.arange(0, db_size, batch_size):
#             images  = db['features'][i:i+batch_size]
#             targets = db['targets'] [i:i+batch_size]
#             yield images, targets

def create_hdf5_generator(db_path, batch_size, keys):
    db = h5py.File(db_path, 'r')
    db_size = db[keys[0]].shape[0]
    while True: 
        for i in np.arange(0, db_size, batch_size):
            res = []
            for key in keys:
                res += [db[key][i:i+batch_size]]
            if len(keys) == 1:
                yield res[0]
            elif len(keys) > 1:
                yield tuple(res)



            
            
def calculate_batch_size(side):

    if side == 256:
        if os.environ["CUDA_VISIBLE_DEVICES"]=="7":
            batch_size = 16
        elif os.environ["CUDA_VISIBLE_DEVICES"] in ["0", "1"]:
            batch_size = 32
            
    if side == 512:
        if os.environ["CUDA_VISIBLE_DEVICES"]=="7":
            batch_size = 8
        elif os.environ["CUDA_VISIBLE_DEVICES"] in ["0", "1"]:
            batch_size = 16
    
    elif side == 1024:
        if os.environ["CUDA_VISIBLE_DEVICES"]=="7":
            batch_size = 4
        elif os.environ["CUDA_VISIBLE_DEVICES"] in ["0", "1"]:
            batch_size = 8
    
    elif side  == 2048:
        if os.environ["CUDA_VISIBLE_DEVICES"] in ["0", "1"]:
            batch_size = 2
        
    if not 'batch_size' in locals():
        print('WARNING: Using default batch_size 2.')
        batch_size = 2
        
    return batch_size



# def initiate_model_old(data_augmentation=False):
    
#     inputs = tf.keras.layers.Input((h_img, w_img, c_feat))

#     # Augmentation
#     if type(data_augmentation) == bool and data_augmentation == False:
#         print('No data augmentation.')
#     elif type(data_augmentation) == keras.engine.sequential.Sequential:
#         inputs = data_augmentation(inputs)
#     else:
#         print('WARNING: Illegal data_augmentation type.')
        
#     # Contraction path
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#     c1 = tf.keras.layers.Dropout(0.1)(c1)
#     c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     b1 = tf.keras.layers.BatchNormalization()(c1)
#     r1 = tf.keras.layers.ReLU()(b1)
#     p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = tf.keras.layers.Dropout(0.1)(c2)
#     c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     b2 = tf.keras.layers.BatchNormalization()(c2)
#     r2 = tf.keras.layers.ReLU()(b2)
#     p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)

#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = tf.keras.layers.Dropout(0.2)(c3)
#     c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     b3 = tf.keras.layers.BatchNormalization()(c3)
#     r3 = tf.keras.layers.ReLU()(b3)
#     p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)

#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = tf.keras.layers.Dropout(0.2)(c4)
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     b4 = tf.keras.layers.BatchNormalization()(c4)
#     r4 = tf.keras.layers.ReLU()(b4)
#     p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)

#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     b5 = tf.keras.layers.BatchNormalization()(c5)
#     r5 = tf.keras.layers.ReLU()(b5)
#     c5 = tf.keras.layers.Dropout(0.3)(r5)
#     c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#     # Expansive path 
#     u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = tf.keras.layers.concatenate([u6, c4])
#     u6 = tf.keras.layers.BatchNormalization()(u6)
#     u6 = tf.keras.layers.ReLU()(u6)

#     u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
#     u7 = tf.keras.layers.concatenate([u7, c3])
#     u7 = tf.keras.layers.BatchNormalization()(u7)
#     u7 = tf.keras.layers.ReLU()(u7)

#     u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
#     u8 = tf.keras.layers.concatenate([u8, c2])
#     u8 = tf.keras.layers.BatchNormalization()(u8)
#     u8 = tf.keras.layers.ReLU()(u8)

#     u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
#     u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
#     u9 = tf.keras.layers.BatchNormalization()(u9)
#     u9 = tf.keras.layers.ReLU()(u9)

#     outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)

#     model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

#     return model


def initiate_model(
    h_img,
    w_img,
    c_feat, 
    num_classes,               
    n_contractions,
    data_augmentation=False):

    depths = [2**(i+4) for i in range(n_contractions)]

    # Input layer
    inputs = tf.keras.layers.Input((h_img, w_img, c_feat))

    # Augmentation
    if type(data_augmentation) == bool and data_augmentation == False:
        print('No data augmentation.')
    elif type(data_augmentation) == keras.engine.sequential.Sequential:
        inputs = data_augmentation(inputs)
    else:
        print('WARNING: Illegal data_augmentation type.')

    # List to store layers (c) for skip connections
    skip_layers = []

    # Contraction path
    p = inputs

    for depth in depths[:-1]:
        # print('depth', depth)
        c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
        c = tf.keras.layers.Dropout(0.1)(c)
        c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        b = tf.keras.layers.BatchNormalization()(c)
        r = tf.keras.layers.ReLU()(b)
        p = tf.keras.layers.MaxPooling2D((2, 2))(r)

        skip_layers += [c] # "push"
        
    # One last contraction block
    depth = depths[-1]
    # print('depth', depth)
    
    c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
    b = tf.keras.layers.BatchNormalization()(c)
    r = tf.keras.layers.ReLU()(b)
    c = tf.keras.layers.Dropout(0.3)(r)
    c = tf.keras.layers.Conv2D(depth, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)

    # skip_layers += [c] # "push"


    # Expansive path 
    u = c
    
    for depth in depths[::-1][1:]:

        # skip_layers = skip_layers[:-1] # "pop"
        u = tf.keras.layers.Conv2DTranspose(depth, (2, 2), strides=(2, 2), padding='same')(u)
        u = tf.keras.layers.concatenate([u, skip_layers[-1]])
        u = tf.keras.layers.BatchNormalization()(u)
        u = tf.keras.layers.ReLU()(u)
        
        skip_layers = skip_layers[:-1] # "pop"
        

    # Sigmoid layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u)

    # Put model together
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # model = tf.keras.Model(inputs=[inputs], outputs=[u])

    return model