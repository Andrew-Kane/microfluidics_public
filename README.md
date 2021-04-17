# microfluidics_public

### developer: Andrew Kane, Ph.D. Student, Denic Laboratory, Harvard University
### email: andrew_kane[at]fas[dot]harvard[dot]edu


## Purpose:
This package was developed to _S. cerevisiae_ mother cells from Z-stack brightfield microscopy images (multipage TIFF format). This implements DeepCell segmentation (https://github.com/CovertLab/DeepCell). It contains a modified version of CellSegment developed by Nicholas Weir (see https://github.com/deniclab/pyto_segmenter) [Weir et al. 2017](https://doi.org/10.1101/136572).

## Installation:
As with any python module, clone the repository into your PYTHONPATH.

### Dependencies:
- python 3.x __Not compatible with Python 2__
- [matplotlib](https://matplotlib.org/)
- [scikit-image](http://scikit-image.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

This module is intended for segmenting _Saccharomyces cerevisiae_ cells during a microfluidics lifespan experiment and assumes that mother cells are roughly centered throughout the entire lifespan and in a single layer. Segmented cells from DeepCell must be cropped down to the mother catcher, which was done using proprietary code provided by Calico Laboratories and is not publically available.

## Contents:

### Segment.py
Classes and methods for identifying whole cells segmented by DeepCell in a Z-stack image, of which each slice is a time point. It does not do well when cells are not present in a single layer. See docstrings for additional details. 
###

Readme last updated 4.17.2021
