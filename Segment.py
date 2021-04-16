"""Classes and methods for generating masks for mother cells from CellStar masks."""

# IMPORT DEPENDENCIES

import os
import sys
import pickle
import time
import re
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import distance_transform_edt

class SegmentObj:
    """ Segmentation data from a DeepCell segmentation image.

    Attributes:
        filename (str): the filename for the original segmentation 
            image.
        raw_img (ndarray): 3D array containing pixel intensities from
            the raw segment image.
        timepoints: the number of timepoints in the raw image.
        height: the height of the raw image.
        width: the width of the raw image.
        threshold (int): the cutoff used for binary thresholding of
            the raw image.
        threshold_img (ndarray): the output from applying the threshold
            to the raw image ndarray.
        filled_img (ndarray): the product of using two-dimensional
            binary hole filling (as implemented in
            scipy.ndimage.morphology) on the thresholded image
        dist_map (ndarray): a 2D euclidean distance transform of the
            filled image.
        smooth_dist_map (ndarray): gaussian smoothed distance map.
        maxima (ndarray): a boolean array indicating the positions of
            local maxima in the smoothed distance map.
        labs (ndarray): an int array with each local maximum in maxima
            assigned a unique integer value, and non-maxima assigned 0.
            required for watershed implementation at the next step.
        watershed_output (ndarray): a 3D array in which each object is
            assigned a unique integer value.
        final_cells (ndarray): the product of cleaning up the filled_cells
            segmented image using a voting filter. see the Segmenter
            reassign_pix_obs method for details.
        mother_cells (ndarray): the product of eliminating all but the 
            mother cell, as determined by what cell is closest to the 
            center of the frame.
        
    """

    def __init__(self, f_directory, filename, raw_img, threshold,
                 threshold_img, filled_img, dist_map, smooth_dist_map,
                 maxima, labs, watershed_output, final_cells,
                 obj_nums, volumes, mother_cells, mother_nums, mother_volumes):
        """Initialize the SegmentObject with segmentation data."""
        print('creating SegmentObject...')
        self.f_directory = f_directory
        self.filename = os.path.basename(filename).lower()
        self.raw_img = raw_img.astype('uint16')
        self.threshold = int(threshold)
        self.threshold_img = threshold_img.astype('uint16')
        self.filled_img = filled_img.astype('uint16')
        self.dist_map = dist_map.astype('uint16')
        self.smooth_dist_map = smooth_dist_map.astype('uint16')
        self.maxima = maxima.astype('uint16')
        self.labs = labs.astype('uint16')
        self.watershed_output = watershed_output.astype('uint16')
        self.final_cells = final_cells.astype('uint16')
        self.slices = int(self.raw_img.shape[0])
        self.height = int(self.raw_img.shape[1])
        self.width = int(self.raw_img.shape[2])
        self.obj_nums =[]
        self.mother_cells = mother_cells.astype('uint16')
        self.mother_nums =[]
        for i in range(len(obj_nums)):
            self.obj_nums.append(obj_nums[i].tolist())
        for i in range(len(mother_nums)):
            self.mother_nums.append(mother_nums[i].tolist())
        self.volumes = volumes
        self.volumes_flag = 'pixels'
        self.pdout = ['volumes']
        self.mother_volumes = mother_volumes
        self.mother_volumes_flag = 'pixels'
        self.pdout = ['mother_volumes']


    def __repr__(self):
        return 'SegmentObj '+ self.filename

    ## PLOTTING METHODS ##
    def plot_raw_img(self,display = False):
        self.plot_stack(self.raw_img, colormap='gray')
        if display:
            plt.show()
    def plot_threshold_img(self, display = False):
        self.plot_stack(self.threshold_img, colormap = 'gray')
        if display:
            plt.show()
    def plot_filled_img(self, display = False):
        self.plot_stack(self.filled_img, colormap = 'gray')
        if display:
            plt.show()
    def plot_dist_map(self, display = False):
        self.plot_stack(self.dist_map)
        if display:
            plt.show()
    def plot_smooth_dist_map(self, display = False):
        self.plot_stack(self.smooth_dist_map)
        if display:
            plt.show()
    def plot_maxima(self, display = False):
        vis_maxima = binary_dilation(self.maxima,
                                     structure = np.ones(shape = (1,5,5)))
        masked_maxima = np.ma.masked_where(vis_maxima == 0, vis_maxima)
        self.plot_maxima_stack(masked_maxima, self.smooth_dist_map)
        if display:
            plt.show()
    def plot_watershed(self, display = False):
        self.plot_stack(self.watershed_output)
        if display:
            plt.show()
    def plot_final_cells(self, display = False):
        self.plot_stack(self.final_cells)
        if display:
            plt.show()

    # OUTPUT METHODS

    def to_csv(self, output_dir = None):
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        for_csv = self.to_pandas()
        for_csv.to_csv(path_or_buf = output_dir + '/' +
                       self.filename[0:self.filename.index('.tif')]+'.csv',
                       index = True, header = True)
    def output_image(self, imageattr, output_dir = None):
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        print('writing image' + str(imageattr))
        io.imsave(str(imageattr)+self.filename, getattr(self,str(imageattr)))

    def output_all_images(self, output_dir = None):
        '''Write all images to a new subdirectory.

        Write all images associated with the SegmentObj to a new
        directory. Name that directory according to the filename of the initial
        image that the object was derived from. This new directory should be a
        subdirectory to the directory containing the original raw image.
        '''
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        print('writing images...')
        io.imsave('raw_'+self.filename, self.raw_img)
        io.imsave('gaussian_'+self.filename, self.gaussian_img)
        io.imsave('filled_threshold_'+self.filename, self.filled_img)
        io.imsave('dist_'+self.filename, self.dist_map)
        io.imsave('smooth_dist_'+self.filename,self.smooth_dist_map)
        io.imsave('maxima_'+self.filename,self.maxima)
        io.imsave('wshed_'+self.filename,self.watershed_output)
        io.imsave('final_cells_'+self.filename, self.final_cells)
    def output_plots(self):
        '''Write PDFs of slice-by-slice plots.

        Output: PDF plots of each image within SegmentObj in a directory
        named for the original filename they were generated from. Plots are
        generated using the plot_stack method and plotting methods defined
        here.
        '''
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.tif')]):
            print('creating output directory...')
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.tif')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.tif')])
        print('saving plots...')
        self.plot_raw_img()
        plt.savefig('praw_'+self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_threshold_img()
        plt.savefig('pthreshold_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_filled_img()
        plt.savefig('pfilled_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_dist_map()
        plt.savefig('pdist_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_smooth_dist_map()
        plt.savefig('psmooth_dist_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_maxima()
        plt.savefig('pmaxima_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_watershed()
        plt.savefig('pwshed_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_final_cells()
        plt.savefig('pfinal_cells_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
    def pickle(self, output_dir = None):
        '''pickle the SegmentObj for later loading.'''
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        with open('pickled_' +
                    self.filename[0:self.filename.index('.tif')] +
                  '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    def output_all(self):
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.tif')]):
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.tif')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.tif')])
        print('outputting all data...')
        self.output_plots()
        self.output_all_images()
        self.pickle()
         # TODO: UPDATE THIS METHOD TO INCLUDE PANDAS OUTPUT
    ## HELPER METHODS ##

    def to_pandas(self):
        '''create a pandas DataFrame of tabulated numeric data.

        the pdout attribute indicates which variables to include in the
        DataFrame.
        '''
        df_dict = {}
        for attr in self.pdout:
            df_dict[str(attr)] = pd.Series(getattr(self, attr))
        if 'volumes' in self.pdout:
            vflag_out = dict(zip(self.obj_nums,
                                 [self.volumes_flag]*len(self.obj_nums)))
            df_dict['volumes_flag'] = pd.Series(vflag_out)
        return pd.DataFrame(df_dict)

    def plot_stack(self, stack_arr, colormap='jet'):
        ''' Create a matplotlib plot with each subplot containing a slice.

        Keyword arguments:
        stack_arr: a numpy ndarray containing pixel intensity values.
        colormap: the colormap to be used when displaying pixel
                  intensities. defaults to jet.

        Output: a pyplot object in which each slice from the image array
                is represented in a subplot. subplots are 4 columns
                across (when 4 or more slices are present) with rows to
                accommodate all slices.
        '''

        nimgs = stack_arr.shape[0] # z axis of array dictates number of slices
        # plot with 4 imgs across

        # determine how many rows and columns of images there are

        if nimgs < 5:
            f, axarr = plt.subplots(1,nimgs)

            for i in range(0,nimgs):
                axarr[i].imshow(stack_arr[i,:,:], cmap=colormap)
                axarr[i].xaxis.set_visible(False)
                axarr[i].yaxis.set_visible(False)
            f.set_figwidth(16)
            f.set_figheight(4)
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT

        else:
            f, axarr = plt.subplots(int(np.ceil(nimgs/4)),4)

            for i in range(0,nimgs):
                r = int(np.floor(i/4))
                c = int(i % 4)
                axarr[r,c].imshow(stack_arr[i,:,:], cmap=colormap)
                axarr[r,c].xaxis.set_visible(False)
                axarr[r,c].yaxis.set_visible(False)

            if nimgs%4 > 0:
                r = int(np.floor(nimgs/4))

                for c in range(nimgs%4,4):
                    axarr[r,c].axis('off')

            f.set_figwidth(16)
            f.set_figheight(4*np.ceil(nimgs/4))
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT
    def plot_maxima_stack(self, masked_max, smooth_dist):

        ''' Creates a matplotlib plot object in which each slice from the image
        is displayed as a single subplot, in a 4-by-n matrix (n depends upon
        the number of slices in the image)'''

        nimgs = masked_max.shape[0] # z axis of array dictates number of slices
        # plot with 4 imgs across

        # determine how many rows and columns of images there are

        if nimgs < 5:
            f, axarr = plt.subplots(1,nimgs)

            for i in range(0,nimgs):
                axarr[i].imshow(smooth_dist[i,:,:], cmap='gray')
                axarr[i].imshow(masked_max[i,:,:], cmap='autumn')
                axarr[i].xaxis.set_visible(False)
                axarr[i].yaxis.set_visible(False)
            f.set_figwidth(16)
            f.set_figheight(4)
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT

        else:
            f, axarr = plt.subplots(int(np.ceil(nimgs/4)),4)

            for i in range(0,nimgs):
                r = int(np.floor(i/4))
                c = int(i%4)
                axarr[r,c].imshow(smooth_dist[i,:,:], cmap='gray')
                axarr[r,c].imshow(masked_max[i,:,:], cmap='autumn')
                axarr[r,c].xaxis.set_visible(False)
                axarr[r,c].yaxis.set_visible(False)

            if nimgs%4 > 0:
                r = int(np.floor(nimgs/4))

                for c in range(nimgs%4, 4):
                    axarr[r,c].axis('off')

            f.set_figwidth(16)
            f.set_figheight(4*np.ceil(nimgs/4))
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT

    ## RESTRUCTURING METHODS ##
    def slim(self):
        '''remove all of the processing intermediates from the object, leaving
        only the core information required for later analysis. primarily
        intended for use when doing batch analysis of multiple images, and
        combining SegmentObj instances with instances of other types of
        objects segmented in a different fluorescence channel.
        '''
        del self.raw_img
        del self.threshold_img
        del self.filled_img
        del self.dist_map
        del self.smooth_dist_map
        del self.maxima
        del self.labs
        del self.watershed_output

        return self
    
class Segmenter:

    def __init__(self, filename, threshold):
        self.filename = filename
        self.threshold = threshold

    def segment(self):
        # start timing
        starttime = time.time()
        # DATA IMPORT AND PREPROCESSING
        f_directory = os.getcwd()
        print('reading ' + self.filename + ' ...')
        raw_img = io.imread(self.filename)
        default_shape = raw_img.shape
        print('raw image imported.')
        # next step's gaussian filter
        print('performing gaussian filtering...')
        gaussian_img = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            temp_img = np.copy(raw_img[i])
            gaussian_img[i] =gaussian_filter(input=raw_img[i], sigma=(2, 2))
        print('cytosolic image smoothed.')
        print('preprocessing complete.')
        # BINARY THRESHOLDING AND IMAGE CLEANUP
        print('thresholding...')
        threshold_img = np.copy(gaussian_img)
        threshold_img[threshold_img < self.threshold] = 0
        threshold_img[threshold_img > 0] = 1
        print('filling holes...')
        filled_img = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            temp_img = np.copy(threshold_img[i])
            filled_img[i] = binary_fill_holes(temp_img)
        print('2d holes filled.')
        print('binary processing complete.')
        # DISTANCE AND MAXIMA TRANFORMATIONS TO FIND CELLS
        print('generating distance map...')
        dist_map = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            temp_img = np.copy(filled_img[i])
            dist_map[i] = distance_transform_edt(temp_img, sampling=(1, 1))
        print('distance map complete.')
        print('smoothing distance map...')
        smooth_dist = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            temp_img = np.copy(dist_map[i])
            smooth_dist[i] = gaussian_filter(temp_img, [4, 4])
        print('distance map smoothed.')
        print('identifying maxima...')
        max_strel_2d = generate_binary_structure(2, 2)
        maxima = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            maxima[i] = maximum_filter(smooth_dist[i],
                                    footprint=max_strel_2d) == smooth_dist[i]
            bgrd_2d = smooth_dist[i] == 0
            eroded_background_2d = binary_erosion(bgrd_2d, structure=max_strel_2d,
                                              border_value=1)
            maxima[i] = np.logical_xor(maxima[i], eroded_background_2d)
        print('maxima identified.')
        # WATERSHED SEGMENTATION
        labs = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            labs[i] = self.watershed_labels(maxima[i])
        print('watershedding...')
        cells = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            cells[i] = watershed(-smooth_dist[i], labs[i], mask=filled_img[i])
        print('raw watershedding complete.')
        print('cleaning up cells...')
        clean_cells = np.zeros(shape=default_shape,dtype='float32')
        for i in range(default_shape[0]):
            clean_cells[i] = self.reassign_pixels_2d(cells[i])
        print('cell cleanup complete.')
        print('SEGMENTATION OPERATION COMPLETE.')
        endtime = time.time()
        runningtime = endtime - starttime
        print('time elapsed: ' + str(runningtime) + ' seconds')
        cell_num=[[] for f in range(default_shape[0])]
        volume=[[] for f in range(default_shape[0])]
        for i in range(default_shape[0]):
            cell_num[i], volume[i] = np.unique(clean_cells[i], return_counts=True)
        for j in range(default_shape[0]):
            cell_num[j] =cell_num[j].astype('uint16')
            volume[j] = volume[j].astype('uint16')
        cell_nums=[[] for f in range(default_shape[0])]
        volumes=[[] for f in range(default_shape[0])]
        for k in range(default_shape[0]):
            volumes[k] = dict(zip(cell_num[k], volume[k]))
            cell_nums[k]=cell_num[k][np.nonzero(cell_num[k])]
        for vol in range(len(volumes)):
            if 0 in volumes[vol]:
                del volumes[vol][0]
        print('determining distances...')
        distances = self.avg_distance_from_center(clean_cells)
        print('distances determined.')
        mother = [[] for f in range(default_shape[0])]
        for j in range(default_shape[0]):
            if volumes[j] == {} or distances[j] == {}:
                pass
            else:
                if self.keywithmaxval(volumes[j]) == self.keywithminval(distances[j]):
                    mother[j] = self.keywithminval(distances[j])
                    print(j)
                else:
                    mother[j] = self.keywithminval(distances[j])
                    print('Maximum size does not agree with minimum distance! Mother may be incorrect, assuming min distance.')
                    print(j)
        mother_cell = np.copy(clean_cells)
        print('eliminating all but mother cell...')
        for frame in range(default_shape[0]):
            mother_cell[frame] = np.copy(clean_cells[frame])
            mother_cell[frame][clean_cells[frame] != mother[frame]] = 0
            mother_cell[frame][mother_cell[frame] > 0] = 1
        print('mothers produced.')
        mother_num = [[] for f in range(default_shape[0])]
        new_volume=[[] for f in range(default_shape[0])]
        for i in range(default_shape[0]):
            mother_num[i], new_volume[i] = np.unique(mother_cell[i], return_counts=True)
        for j in range(default_shape[0]):
            mother_cell[j] = mother_cell[j].astype('uint16')
            new_volume[j] = new_volume[j].astype('uint16')
        mother_nums=[[] for f in range(default_shape[0])]
        new_volumes=[[] for f in range(default_shape[0])]
        for k in range(default_shape[0]):
            new_volumes[k] = dict(zip(mother_num[k], new_volume[k]))
            mother_nums[k]=mother_num[k][np.nonzero(mother_num[k])]
        for vol in range(len(new_volumes)):
            del new_volumes[vol][0]
        return SegmentObj(f_directory, self.filename, raw_img, self.threshold,
                              threshold_img, filled_img, dist_map,
                              smooth_dist, maxima, labs, cells,
                              clean_cells, cell_nums, volumes, mother_cell, mother_nums,
                              new_volumes)

    def watershed_labels(self, maxima_img):
        '''Takes a boolean array with maxima labeled as true pixels
        and returns an array with maxima numbered sequentially.'''

        max_y, max_x = np.nonzero(maxima_img)

        label_output = np.zeros(maxima_img.shape)

        for i in range(0,len(max_y)):
            label_output[max_y[i],max_x[i]] = i+1

        return(label_output)

    def reassign_pixels_2d(self, watershed_img_2d):
        '''In 2D, go over each segmented pixel in the image with a
        structuring element. Measure the frequency of each different
        pixel value that occurs within this structuring element (i.e. 1
        for one segmented cell, 2 for another, etc. etc.). Generate a
        new image with each pixel value set to that of the most common
        non-zero (non-background) pixel within its proximity. After
        doing this for every pixel, check and see if the new image is
        identical to the old one. If it is, stop; if not, repeat with
        the new image as the starting image.'''

        #retrieve number of cells in the watershedded image
        ncells = np.unique(watershed_img_2d).size-1
        if ncells == 0:
            return watershed_img_2d
        else:
            strel = np.array(
                [[0,0,0,1,0,0,0],
                 [0,0,1,1,1,0,0],
                 [0,1,1,1,1,1,0],
                 [1,1,1,1,1,1,1],
                 [0,1,1,1,1,1,0],
                 [0,0,1,1,1,0,0],
                 [0,0,0,1,0,0,0]])
            old_img = watershed_img_2d
            mask_y,mask_x = np.nonzero(old_img)
            tester = 0
            counter = 0
            while tester == 0:
                new_img = np.zeros(old_img.shape)
                for i in range(0,len(mask_y)):
                    y = mask_y[i]
                    x = mask_x[i]
                    # shift submatrix if the pixel is too close to the edge to be
                    # centered within it
                    if y-3 < 0:
                        y = 3
                    if y+4 >= old_img.shape[0]:
                        y = old_img.shape[0] - 4
                    if x-3 < 0:
                        x = 3
                    if x+4 >= old_img.shape[1]:
                        x = old_img.shape[1]-4
                    # take a strel-sized matrix around the pixel of interest.
                    a = old_img[y-3:y+4,x-3:x+4]
                    # mask for the pixels that I'm interested in comparing to
                    svals = np.multiply(a,strel)
                    # convert this to a 1D array with zeros removed
                    cellvals = svals.flatten()[np.nonzero(
                            svals.flatten())].astype(int)
                    # count number of pixels that correspond to each
                    # cell in proximity
                    freq = np.bincount(cellvals,minlength=ncells)
                    # find cell with most abundant pixels in vicinity
                    top_cell = np.argmax(freq)
                    # because argmax only returns the first index if there are
                    # multiple matches, I need to check for duplicates. if
                    # there are duplicates, I'm going to leave the value as the
                    # original, regardless of whether the set of most abundant
                    # pixel values contain the original pixel (likely on an
                    # edge) or not (likely on a point where three cells come
                    # together - this is far less common)
                    if np.sum(freq==freq[top_cell]) > 1:
                        new_img[mask_y[i],mask_x[i]] = old_img[mask_y[i],mask_x[i]]
                    else:
                        new_img[mask_y[i],mask_x[i]] = top_cell

                if np.array_equal(new_img,old_img):
                    tester = 1
                else:
                    old_img = np.copy(new_img)
                counter += 1
            print('    number of iterations = ' + str(counter))
            return new_img
    
    def avg_distance_from_center(self, shape_array):
        center_point = center_point =(shape_array.shape[1]/2.+15,shape_array.shape[2]/2.)
        dists=[[] for f in range(shape_array.shape[0])]
        for frame in range(shape_array.shape[0]):
            cellnums = np.unique(shape_array[frame]).tolist()
            cellnums.pop(0)
            cells = []
            dist = []
            for cell in cellnums:
                shape_pixels = np.where(shape_array[frame] == cell)
                total_distance = 0.
                for i in range(len(shape_pixels[0])):
                    i_ind = float(shape_pixels[0][i])
                    j_ind = float(shape_pixels[1][i])
                    total_distance += ((i_ind - center_point[0])**2.0 + (j_ind - center_point[1])**2.0)**0.5
                avg_distance = total_distance / len(shape_pixels[0])
                cells.append(cell)
                dist.append(avg_distance)
            distances = dict(zip(cells,dist))
            dists[frame] = distances
        return dists
    
    def keywithmaxval(self, d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]
    
    def keywithminval (self, d):
        """ a) create a list of the dict's keys and values;
            b) return the key with the min value"""
        v=list(d.values())
        print(v)
        k=list(d.keys())
        return k[v.index(min(v))]


if __name__ == '__main__':
    ''' Run segmentation on all images in the working directory.

    sys.argv contents:
        threshold: a number that will be used as the threshold to binarize
        the cytosolic fluorescence image.
    '''

    threshold = int(sys.argv[1])
    wd_contents = os.listdir(os.getcwd())
    imlist = []
    for f in wd_contents:
        if (f.endswith('.tif') or f.endswith('.tiff')
            or f.endswith('.TIF') or f.endswith('.TIFF')):
            imlist.append(f)
    print('final imlist:')
    print(imlist)
    for i in imlist:
        print('initializing segmenter object...')
        i_parser = Segmenter(i,threshold)
        print('initializing segmentation...')
        i_obj = i_parser.segment()
        i_obj.output_all()