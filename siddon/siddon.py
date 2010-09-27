"""
A module which performs tomographic projections and backprojections.
The projections performed by Siddon are 3-dimensional conic
projections.

Parameters to the projection makes partly use of the FITS standard:
http://archive.stsci.edu/fits/fits_standard/.

FITS files are heavily used in astrophysics and can store metadata of
any kind along data.

The mandatory keywords are as follows:

- common parameters

  * NAXIS : Number of dimensions.

  * NAXIS{1,2,[3]} : Array shape.

  * BITPIX : fits convention for data types. Correspondances:

       8    np.int8
      16    np.int16
      32    np.int32
     -32    np.float32
     -64    np.float64

  * CRPIX{1,2,[3]} : position of the reference pixel in pixels (can a be
    float).

  * CRVAL{1,2,[3]} : position of the reference pixel in physical
    coordinates.

  * CDELT{1,2,[3]} : size of a pixel in physical coordinates

- Cubic maps :

  There is no extra parameters for the cubic maps.

- Images :

  * D: Distance between the viewpoint and the map reference pixel.

  * DX, DY, DZ: Coordinates of the viewpoints relative to the
    reference pixel of the cube.

  * LON, LAT, ROL: longitude, latitude, roll. Defines the position and
    orientation of the viewpoint.

The parameters need to be stored in the header dict of a FitsArray.
For the set of images, all the images are concatenated in a single
array forming a cube. The metadata is stored in an InfoArray header.
Each value of the header dict is a 1d ndarray of length the number of
images.

"""
import numpy as np
import time
import copy
import os
import pyfits
import fitsarray as fa
from parse_templates import siddon_dict_list, suffix_str, ctypes_inv, obstacles_inv
for siddon_dict in siddon_dict_list:
    exec_str = "from _C_siddon"
    exec_str += suffix_str
    exec_str += " import siddon as siddon"
    exec_str += suffix_str
    exec(exec_str % siddon_dict)
    del exec_str

# projector
def projector(data, cube, obstacle=None):
    """
    Project a cubic map into a data cube using the Siddon algorithm.
    """
    cube.header = dict(cube.header)
    if data.dtype != cube.dtype:
        raise ValueError("data and cube map should have the same data-type")
    my_siddon_dict = {"ctype":ctypes_inv[data.dtype.name],
                      "obstacle":obstacles_inv[obstacle]}
    proj_str = "siddon" + suffix_str + "(data, cube, 0)"
    exec(proj_str % my_siddon_dict)
    return data

def backprojector(data, cube, obstacle=None):
    """
    Backproject a data cube into a cubic map using the Siddon algorithm.
    """
    cube.header = dict(cube.header)
    if data.dtype != cube.dtype:
        raise ValueError("data and cube map should have the same data-type")
    my_siddon_dict = {"ctype":ctypes_inv[data.dtype.name],
                      "obstacle":obstacles_inv[obstacle]}
    proj_str = "siddon" + suffix_str + "(data, cube, 1)"
    exec(proj_str % my_siddon_dict)
    return cube

def dataarray_from_header(header):
    """
    Output an InfoArray using a list of headers.
    """
    shape = [int(header['NAXIS' + str(i + 1)][0])
             for i in xrange(int(header['NAXIS'][0]))]
    shape += len(header['NAXIS']),
    dtype = fa.bitpix[str(int(header['BITPIX'][0]))]
    return fa.InfoArray(shape, header=header, dtype=dtype)

def update_header(array):
    """
    Update header to add siddon required keywords and convert to
    appropriate units.
    """
    instrume = array.header.get('INSTRUME')
    if instrume is None:
        raise ValueError('array header does not have an INSTRUME keyword')
    if instrume == 'SECCHI':
        siddon.secchi.update_header(array)
    else:
        raise ValueError('Instrument ' + instrume + ' not yet handled')
