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
import lo
from _C_siddon import siddon_sun as C_siddon_sun
from _C_siddon import siddon as C_siddon

# projector
def projector(data, cube):
    """
    Project a cubic map into a data cube using the Siddon algorithm.
    """
    cube.header = dict(cube.header)
    C_siddon(data, cube, 0)
    return data

def backprojector(data, cube):
    """
    Backproject a data cube into a cubic map using the Siddon algorithm.
    """
    cube.header = dict(cube.header)
    C_siddon(data, cube, 1)
    return cube

def siddon_lo(data_header, cube_header):
    """
    A linear operator performing projection and backprojection using
    Siddon.
    """
    data = dataarray_from_header(data_header)
    data[:] = 0
    cube = fa.fitsarray_from_header(cube_header)
    cube[:] = 0
    def matvec(x):
        y = dataarray_from_header(data_header)
        y[:] = 0
        projector(y, x)
        return y
    def rmatvec(x):
        y = fa.fitsarray_from_header(cube_header)
        y[:] = 0
        backprojector(x, y)
        return y
    return lo.ndsubclass(cube, data, matvec=matvec, rmatvec=rmatvec)

def projector_sun(data, cube):
    """
    Project a cubic map into a data cube using the Siddon algorithm.
    Rays are blocked at a centered sphere of radius one (as it is the
    case in solar tomography).
    """
    cube.header = dict(cube.header)
    C_siddon_sun(data, cube, 0)
    return data

def backprojector_sun(data, cube):
    """
    Backproject a data cube into a cubic map using the Siddon algorithm.
    Rays are blocked at a centered sphere of radius one (as it is the
    case in solar tomography).
    """
    cube.header = dict(cube.header)
    C_siddon_sun(data, cube, 1)
    return cube

def siddon_sun_lo(data_header, cube_header):
    """
    A linear operator performing projection and backprojection using
    Siddon.
    Rays are blocked at a centered sphere of radius one (as it is the
    case in solar tomography).
    """
    data = dataarray_from_header(data_header)
    data[:] = 0
    cube = fa.fitsarray_from_header(cube_header)
    cube[:] = 0
    def matvec(x):
        y = dataarray_from_header(data_header)
        y[:] = 0
        projector_sun(y, x)
        return y
    def rmatvec(x):
        y = fa.fitsarray_from_header(cube_header)
        y[:] = 0
        backprojector_sun(x, y)
        return y
    return lo.ndsubclass(cube, data, matvec=matvec, rmatvec=rmatvec)

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
