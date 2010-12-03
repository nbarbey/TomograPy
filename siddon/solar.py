"""
Generic code for WCS compatible data.
"""
import os
import time
import pyfits
import numpy as np
import fitsarray as fa

# constants
solar_radius = 695000000. # in m
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
    files = list()
    for fname in fnames:
        try:
            files.append(pyfits.fitsopen(os.path.join(path, fname))[0])
        except(IOError):
            pass
    files = filter_files(files, **kargs)
    for i, f in enumerate(files):
        fits_array = fa.hdu2fitsarray(f)
        if bin_factor is not None:
            fits_array = fits_array.bin(bin_factor)
        update_header(fits_array)
        fits_array = fits_array.T
        if i == 0:
            data = fa.InfoArray(fits_array.shape + (len(files),), header=dict(fits_array.header))
            for k in data.header.keys():
                if np.isscalar(data.header[k]):
                    data.header[k] = np.asarray((data.header[k], ))
        data[..., i] = fits_array
        if i != 0:
            for k in fits_array.header.keys():
                if np.isscalar(fits_array.header[k]):
                    data.header[k] = np.concatenate([data.header[k], np.asarray((fits_array.header[k], ))])
                else:
                    data.header[k] = np.concatenate([data.header[k], fits_array.header[k]])
    data = data.astype(dtype)
    return data

def update_header(array):
    # read useful keywords
    lon = np.radians(array.header['CRLN_OBS'])
    lat = np.radians(array.header['CRLT_OBS'])
    # roll angle
    try:
        pc2_1 = array.header['PC2_1']
        pc1_1 = array.header['PC1_1']
        cdelt1 = array.header['CDELT1']
        cdelt2 = array.header['CDELT2']
        rol =np.arctan(pc2_1 * cdelt2 / (pc1_1 * cdelt1))
    except(KeyError):
        rol = np.radians(array.header['CROTA2'])
    # distance from observer to sun center
    try:
        d = array.header['DSUN_OBS'] / solar_radius
    except(KeyError):
        try:
            x = array.header['HAEX_OBS']
        except(KeyError):
            x = array.header['HAEX']
        try:
            y = array.header['HAEY_OBS']
        except(KeyError):
            y = array.header['HAEY']
        try:
            z = array.header['HAEZ_OBS']
        except(KeyError):
            z = array.header['HAEZ']
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) / solar_radius
    # infere linked values
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
    # optionnaly remove Z
    time_str.rstrip("Z")
    dpos = time_str.rfind(".")
    # remove fraction of seconds and add them afterwards
    time_str, sec_float = time_str[:dpos], float(time_str[dpos:])
    format = '%Y-%m-%dT%H:%M:%S'
    current_time = time.strptime(time_str, format)
    current_time = time.mktime(current_time)
    current_time += sec_float
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

def define_data_mask(data, data_rmin=None, data_rmax=None, 
                     mask_negative=False, mask_nan=True, **kwargs):
    """
    Defines a mask of shape data.shape.

    Inputs
    ------

    data: A data InfoArray with 'RSUN' and 'CRPIX{1,2}' as metadata.
      The data set.

    data_rmin: float (optional)
      Data below data_rmin is masked. data_rmin is defined relatively to RSUN

    data_rmax: float (optional)
      Data above data_rmax is masked. data_rmax is defined relatively to RSUN

    mask_negative: boolean
      If True, negative data is masked.

    mask_nan: boolean
      If True, nan data is masked.

    Output
    ------
    data_mask: ndarray of booleans of shapa data.shape
    """
    data_mask = np.zeros(data.shape, dtype=bool)
    # if no radius limits no need to compute R
    if data_rmin is not None or data_rmax is not None:
        R = distance_to_sun_center(data)
        if data_rmin is not None:
            data_mask[(R < data_rmin)] = 1
        if data_rmax is not None:
            data_mask[(R > data_rmax)] = 1
    if mask_negative:
        data_mask[data < 0.] = 1
    if mask_nan:
        data_mask[np.isnan(data)] = 1
    return data_mask

def distance_to_sun_center(data):
    """
    Outputs an array containing the distance from the Sun center in
    the data.

    Inputs
    ------

    data: A data InfoArray with 'RSUN' and 'CRPIX{1,2}' as metadata.
      The data set.

    Output
    ------
    R: ndarray of shapa data.shape
      The distance to the Sun center on the images in % of RSUN.
      RSUN is the radius of the Sun on one image in number of pixels.
    """
    R = np.zeros(data.shape)
    Rsun = compute_rsun(data)
    # loop on images
    for i in xrange(data.shape[-1]):
        # get axes
        crpix1 = data.header['CRPIX1'][i]
        crpix2 = data.header['CRPIX2'][i]
        y = (np.arange(data.shape[0]) - crpix1) / Rsun[i]
        x = (np.arange(data.shape[0]) - crpix2) / Rsun[i]
        # generate 2D repeated axes
        X, Y = np.meshgrid(x, y)
        # radius computation
        R[..., i] = np.sqrt(X ** 2 + Y ** 2)
    return R

def compute_rsun(data):
    rsun = np.empty(data.shape[-1])
    for i in xrange(data.shape[-1]):
        d = data.header['D'][i]
        cdelt1 = data.header['CDELT1'][i]
        cdelt2 = data.header['CDELT2'][i]
        if cdelt1 != cdelt2:
            raise ValueError('Meaningless if cdelts are not equal.')
        rsun[i] = np.arctan(1. / d) / cdelt1
    return rsun

def define_map_mask(cube, obj_rmin=None, obj_rmax=None, **kwargs):
    """
    Output a mask of shape cube.shape.
    """
    obj_mask = np.zeros(cube.shape, dtype=bool)
    if obj_rmin is not None or obj_rmax is not None:
        R = map_radius(cube)
        if obj_rmin is not None:
            obj_mask[R < obj_rmin] = 1
        if obj_rmax is not None:
            obj_mask[R > obj_rmax] = 1
    return obj_mask

def map_radius(cube):
    """
    Outputs a cube containing the distance to the Sun center in a map
    cube.
    """
    R = np.zeros(cube.shape)
    x, y, z = cube.axes()
    X, Y = np.meshgrid(x, y)
    for i, zt in enumerate(z):
        R[..., i] = np.sqrt(X ** 2 + Y ** 2 + zt ** 2)
    return R

def slice_data(data, s):
    """
    Slices data InfoArray.
    """
    sd = 2 * (slice(None, None, None), ) + (s,)
    out = data[sd]
    # copy header elements as it is not done usually
    out.header = data.header.copy()
    for k in data.header.keys():
        out.header[k] = data.header[k][s].copy()
    return out

def concatenate(data_list):
    out = np.concatenate(data_list, axis=-1)
    # copy header and key values
    header = data_list[0].header.copy()
    for k in header.keys():
        header[k] = data_list[0].header[k].copy()
    # concatenate key values
    for k in header.keys():
        header[k] = np.concatenate([d.header[k] for d in data_list])
    out = fa.asinfoarray(out, header)
    return out

def sort_data_array(data):
    # sort in time
    times = [convert_time(t) for t in data.header['DATE_OBS']]
    ind = np.argsort(times)
    data_list = []
    for i in ind:
        s = slice(i, i + 1)
        data_list.append(slice_data(data, s))
    data = concatenate(data_list)
    return data

def temporal_groups_indexes(data, dt_min):
    # XXX buggy if no groups !!!
    times = [convert_time(t) for t in data.header['DATE_OBS']]
    ind1 = list(np.where(np.diff(times) < dt_min)[0])
    return ind1

def temporal_groups(data, dt_min):
    """
    Generates a list of data InfoArray regrouped by time.
    All images closer in time to dt_min will be in the same array.
    """
    ind1 = temporal_groups_indexes(data, dt_min)
    ind2 = ind1[1:] + [None,]
    return [slice_data(data, slice(i, j, None)) for i, j in zip(ind1, ind2)]
