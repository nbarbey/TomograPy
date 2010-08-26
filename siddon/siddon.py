"""A module which handles tomographic reconstruction cube and data
"""
import numpy as np
import time
import copy
import os
import pyfits
import fitsarray as fa
import lo
from _C_siddon import siddon as C_siddon

# constants
solar_radius = 695000 # in km
arcsecond_to_radian = np.pi/648000 #pi/(60*60*180)

# projector
def projector(data, cube):
    data[:] = data.astype('float32')
    cube[:] = cube.astype('float32')
    for k in data.header:
        if data.header[k].dtype == np.dtype('float64'):
            data.header[k] = data.header[k].astype('float32')
    for k in cube.header:
        cube.header[k] = np.float32(cube.header[k])
    cube.header = dict(cube.header)
    C_siddon(data, cube, 0)
    return data

def backprojector(data, cube):
    data[:] = data.astype('float32')
    cube[:] = cube.astype('float32')
    for k in data.header:
        if data.header[k].dtype == np.dtype('float64'):
            data.header[k] = data.header[k].astype('float32')
    for k in cube.header:
        cube.header[k] = np.float32(cube.header[k])
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
        projector(y, x.astype(np.float32))
        return y
    def rmatvec(x):
        y = fa.fitsarray_from_header(cube_header)
        y[:] = 0
        backprojector(x.astype(np.float32), y)
        return y
    return lo.ndsubclass(cube, data, matvec=matvec, rmatvec=rmatvec)

def dataarray_from_header(header):
    shape = [int(header['NAXIS' + str(i + 1)][0])
             for i in xrange(int(header['NAXIS'][0]))]
    shape += len(header['NAXIS']),
    dtype = fa.bitpix[str(int(header['BITPIX'][0]))]
    return fa.InfoArray(shape, header=header, dtype=dtype)

# data handling
def read_secchi_data(path, dtype=np.float32, bin_factor=None, **kargs):
    """
    Read SOHO / STEREO data files and output a Data instance

    Input :

      path : path of the data set

      dtype : cast of the data array
      
      kargs : arguments of the data filtering
    """
    if not os.path.isdir(path):
        raise ValueError('Directory does not exist')
    # read files
    fnames = os.listdir(path)
    files = [pyfits.fitsopen(os.path.join(path, fname))[0] for fname in fnames]
    files = filter_files(files, **kargs)
    fits_arrays = list()
    for f in files:
        fits_array = fa.hdu2fitsarray(f)
        if bin_factor is not None:
            fits_array = fits_array.bin(bin_factor)
        update_header(fits_array)
        fits_arrays.append(fits_array)
    data = fa.infoarrays2infoarray(fits_arrays)
    data = data.astype(dtype)
    return data

def update_header(array):
    """Update header to add siddon required keywords
    and convert to appropriate units
    """
    instrume = array.header.get('INSTRUME')
    if instrume is None:
        raise ValueError('array header does not have an INSTRUME keyword')
    if array.header['INSTRUME'] == 'SECCHI':
        secchi_update_header(array)

# SECCHI specific code
def secchi_update_header(array):
    # read useful keywords
    lon = array.header['HEL_LON']
    lat = array.header['HEL_LAT']
    rol = np.radians(array.header['SC_ROLL'])
    x = array.header['HEC_X']
    y = array.header['HEC_Y']
    z = array.header['HEC_Z']
    # infere linked values
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2) / solar_radius
    xd = d * np.cos(lat) * np.cos(lon)
    yd = d * np.cos(lat) * np.sin(lon)
    zd = d * np.sin(lat)
    # update the header
    array.header.update('lon', lon)
    array.header.update('lat', lat)
    array.header.update('rol', rol)
    array.header.update('d', d)
    array.header.update('xd', xd)
    array.header.update('yd', yd)
    array.header.update('zd', zd)
    # convert to radians
    array.header['CDELT1'] *= arcsecond_to_radian
    array.header['CDELT2'] *= arcsecond_to_radian
    # others
    time_str = array.header['DATE_OBS']
    array.header.update('time', secchi_convert_time(time_str))

def secchi_convert_time(time_str):
    time_str = time_str[:-4]
    format = '%Y-%m-%dT%H:%M:%S'
    current_time = time.strptime(time_str, format)
    current_time = time.mktime(current_time)
    return current_time

def filter_files(files, instrume=None, obsrvtry=None, detector=None, 
                 time_window=None, time_step=None):
    out_files = list()
    # sort list by time
    files.sort(cmp=time_compare)
    # if a time_window is given convert it to float
    if time_window is not None:
        time_min = secchi_convert_time(time_window[0])
        time_max = secchi_convert_time(time_window[1])
        # enforce order
        if time_min > time_max:
            time_min, time_max = time_max, time_min
    if time_step is not None:
        if isinstance(time_step, str):
            time_step_val = secchi_convert_time(time_step)
        else:
            time_step_val = time_step
    for i, f in enumerate(files):
        bad = 0
        # check for origin of data
        if instrume is not None:
            if instrume != f.header['INSTRUME']:
                bad = 1
        if obsrvtry is not None:
            if obsrvtry != f.header['OBSRVTRY']:
                bad = 1
        if detector is not None:
            if detector != f.header['DETECTOR']:
                bad = 1
        # check for time window
        if time_window is not None:
            time_str = f.header['DATE_OBS']
            time_val = secchi_convert_time(time_str)
            if (time_val > time_max) or (time_val < time_min):
                bad = 1
        # check for time step
        if time_step is not None:
            time_str = f.header['DATE_OBS']
            time_val = secchi_convert_time(time_str)
            if len(out_files) != 0:
                last_time_str = out_files[-1].header['DATE_OBS']
                last_time_val = secchi_convert_time(last_time_str)
                if abs(last_time_val - time_val) < time_step_val:
                    bad = 1
        if bad == 0:
            out_files.append(f)
    return out_files

def time_compare(x, y):
    a = secchi_convert_time(x.header['DATE_OBS'])
    b = secchi_convert_time(y.header['DATE_OBS'])
    if a > b:
        return 1
    elif a == b:
        return 0
    else: # a < b
        return -1

