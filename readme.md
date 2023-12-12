# Goal

Detect small objects, like chromosomes, in images.

# TO DO

Data augmentation, now as the first layer of the U-Net, is only applied to the features, not to the target image.  This is problematic for horizontal and vertical flips and for rotations.  This can be fixed by creating a data generator that includes flipping and rotating.
