import os
from skimage.feature import peak_local_max
from scipy.signal import fftconvolve
import numpy as np
from skimage.external.tifffile import tifffile 


def match_template_to_image(image, template, min_distance=25, threshold_rel=0.35):
    """ Uses fourier transforms to find peaks in image and locate catchers
    
        image -- 2d np array in which to look for catchers
        template -- 2d array with example of catcher -- cropped tight around catcher
        returns a list of array of coordinates where template occured in the image"""
    return peak_local_max(fftconvolve(image - np.mean(image),
                                      template[::-1, ::-1] -
                                      np.mean(template), 'same'),
                          min_distance=min_distance, 
                          threshold_rel=threshold_rel)
    
def find_catchers(timepoint,
                  pos,
                  name_format, # (pos, channel, time, filetype)
                  dest_parent_dir,
                  experiment_directory,
                  channel = "BF",
                  channel_number = 1):
    '''Finds catchers in image and writes to new file
    
        timepoint -- the timepoint to find catchers in
        pos -- the position to find catchers in
        name_format -- the name format to search for files
        dest_parent_dir -- directory to write out catcher locations
        experimental directory -- directory from which to pull data
        channel -- channel to search for catchers, default is BF
        channel_number -- channel number to search for catchers, default is 1'''
    tif_path = os.path.join(experiment_directory,
                            'Pos%d' % pos,
                             channel,
                             name_format % (pos, "c%d%s" % (channel_number, channel), timepoint))
    tif = tifffile.imread([tif_path])
    if len(tif.shape) > 2:
        tif = np.max(tif, axis=0)
    blobs = match_template_to_image(image=tif[0:512, 0:512],
                                    template=cropped_reference_catcher,
                                    min_distance=100,
                                    threshold_rel=0.20)
    catcher_loc_path = os.path.join(
        dest_parent_dir,
        (name_format.split(".")[0]+".catcher_loc") %(pos, 
                                    "c%d%s" % (channel_number, channel), 
                                    timepoint))
    with open(catcher_loc_path, "w") as catcher_locs_fout:
        catcher_locs_fout.write('%s\t%s\n' % ('catcher_x_coord', 'catcher_y_coord'))
        for blob in blobs:
            catcher_locs_fout.write('%s\t%s\n' % (blob[1], blob[0]))
    return (tif, blobs)