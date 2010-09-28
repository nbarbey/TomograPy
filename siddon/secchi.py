"""
Code specific to the SECCHI instrument of the STEREO mission
"""
import os
import time
import pyfits
import numpy as np
import fitsarray as fa

# constants
solar_radius = 695000 # in km
arcsecond_to_radian = np.pi/648000 #pi/(60*60*180)

# data handling
def read_data(path, dtype=np.float64, bin_factor=None, **kargs):
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
    array.header.update('time', convert_time(time_str))

def convert_time(time_str):
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
        time_min = convert_time(time_window[0])
        time_max = convert_time(time_window[1])
        # enforce order
        if time_min > time_max:
            time_min, time_max = time_max, time_min
    if time_step is not None:
        if isinstance(time_step, str):
            time_step_val = convert_time(time_step)
        else:
            time_step_val = time_step
    for i, f in enumerate(files):
        bad = 0
        # check for origin of data
        if instrume is not None:
            if f.header['INSTRUME'] not in instrume:
                bad = 1
        if obsrvtry is not None:
            if f.header['OBSRVTRY'] not in obsrvtry:
                bad = 1
        if detector is not None:
            if f.header['DETECTOR'] not in detector:
                bad = 1
        # check for time window
        if time_window is not None:
            time_str = f.header['DATE_OBS']
            time_val = convert_time(time_str)
            if (time_val > time_max) or (time_val < time_min):
                bad = 1
        # check for time step
        if time_step is not None:
            time_str = f.header['DATE_OBS']
            time_val = convert_time(time_str)
            if len(out_files) != 0:
                last_time_str = out_files[-1].header['DATE_OBS']
                last_time_val = convert_time(last_time_str)
                if abs(last_time_val - time_val) < time_step_val:
                    bad = 1
        if bad == 0:
            out_files.append(f)
    return out_files

def time_compare(x, y):
    a = convert_time(x.header['DATE_OBS'])
    b = convert_time(y.header['DATE_OBS'])
    if a > b:
        return 1
    elif a == b:
        return 0
    else: # a < b
        return -1
