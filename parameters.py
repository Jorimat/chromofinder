# folderData = '/data/home/jorism/data/sedum_some_tagged/camera_158/2022-10-03T14/jpgs/'
# folderData = '/media/joris/OS/data/kiwibes2023/camera_158/2023-06-05T12'
# folderData = '/data/home/jorism/data/kiwibes2023-camera_158-2023-06-05T12-tagged/'



# side = 256
# side = 512
side = 1024
# side = 2048

w_img = side
h_img = side

num_classes = 1

# n_contractions = 1
# n_contractions = 2
# n_contractions = 3
# n_contractions = 4
# n_contractions = 5
# n_contractions = 6
# n_contractions = 7
# n_contractions = 8
# n_contractions = 9

n_contractions = 11


folderData        = '/data/home/jorism/data/kiwibes2023/camera_158/2023-06-05T12/'

folderHDF5        = f'/data/home/jorism/data/kiwibes2023-unet-006-{w_img}x{h_img}'
folderModel       = f'/data/home/jorism/models/kiwibes2023-unet-006-{w_img}x{h_img}-{n_contractions}_contractions'
folderPredictions = f'/data/home/jorism/data/kiwibes2023-unet-006-{w_img}x{h_img}-{n_contractions}_contractions_prediction'
folderViz         = f'/data/home/jorism/data/kiwibes2023-unet-006-{w_img}x{h_img}-{n_contractions}_contractions_visualisations'


n_pos_crops_per_image = 2
n_neg_crops_per_image = 2
frac_train = 0.8
max_n_files = 1000
# max_n_files = 20

# batch_size = 4

diff_scheme = 'central'
epochs=25

verbose = False
visualizePredictions = True

split_type = 'longitudinal'

    
fileModel    = f'{folderModel}/model.hdf5'
fileHDFTrain = f'{folderHDF5}/train.hdf5'
fileHDFTest  = f'{folderHDF5}/test.hdf5'
fileHDFTestPredictions = f'{folderPredictions}/test.hdf5'