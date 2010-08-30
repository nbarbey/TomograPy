"""A module which handles tomographic reconstruction cube and data
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
    cube.header = dict(cube.header)
    C_siddon(data, cube, 0)
    return data

def backprojector(data, cube):
    cube.header = dict(cube.header)
    C_siddon(data, cube, 1)
    return cube

def siddon_lo(data_header, cube_header):
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
    cube.header = dict(cube.header)
    C_siddon_sun(data, cube, 0)
    return data

def backprojector_sun(data, cube):
    cube.header = dict(cube.header)
    C_siddon_sun(data, cube, 1)
    return cube

def siddon_sun_lo(data_header, cube_header):
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
    shape = [int(header['NAXIS' + str(i + 1)][0])
             for i in xrange(int(header['NAXIS'][0]))]
    shape += len(header['NAXIS']),
    dtype = fa.bitpix[str(int(header['BITPIX'][0]))]
    return fa.InfoArray(shape, header=header, dtype=dtype)

def update_header(array):
    """Update header to add siddon required keywords
    and convert to appropriate units
    """
    instrume = array.header.get('INSTRUME')
    if instrume is None:
        raise ValueError('array header does not have an INSTRUME keyword')
    if array.header['INSTRUME'] == 'SECCHI':
        siddon.secchi.update_header(array)
