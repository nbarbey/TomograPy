"""
Regroup functions to easily perform simulations.
"""
import copy
import numpy as np
import siddon
import fitsarray as fa

default_image_dict = {'NAXIS':2, 'NAXIS1':1, 'NAXIS2':1,
                      'BITPIX':-64, 'SIMPLE':True,
                      'CRPIX1':0.5, 'CRPIX2':0.5, 'CDELT1':1., 'CDELT2':1.,
                      'LON':0., 'LAT':0., 'ROL':0.,
                      'D':1., 'M1':1., 'M2':0., 'M3':0.}
image_keys = default_image_dict.keys()
default_image = fa.fitsarray_from_header(default_image_dict)

default_object_dict = {'NAXIS':3, 'NAXIS1':1, 'NAXIS2':1, 'NAXIS3':1,
                       'BITPIX':-64, 'SIMPLE':True,
                       'CRPIX1':0.5, 'CRPIX2':0.5, 'CRPIX3':0.5,
                       'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,
                       'CDELT1':1., 'CDELT2':1., 'CDELT3':1., }
object_keys = default_object_dict.keys()
default_object = fa.fitsarray_from_header(default_object_dict)

class Image(fa.InfoArray):
    """
    A subclass of FitsArray with mandatory keywords defining an image
    """
    def __new__(subtype, shape, data=None, dtype=float, buffer=None, offset=0,
                strides=None, order=None, header=None):
        if header is None:
            header = copy.copy(default_image_dict)
        obj = fa.InfoArray.__new__(subtype, shape=shape, data=data, 
                                   dtype=dtype, buffer=buffer, offset=offset,
                                   strides=strides, order=order, header=header)
        # look for mandatory keywords
        for k in image_keys:
            if not obj.header.has_key(k):
                obj.header[k] = default_image_dict[k]
        return obj

    def update(self, key, value):
        self.header[key] = value
        if key == 'LAT' or key == 'LON' or key == 'D':
            self._update_from_spherical(key, value)
        elif key == 'M1' or key == 'M2' or key == 'M3':
            self._update_from_cartesian(key, value)

    def _update_from_spherical(self, key, value):
        lon = self.header['LON']
        lat = self.header['LAT']
        d = self.header['D']
        self.header['M1'] = d * np.cos(lat) * np.cos(lon)
        self.header['M2'] = d * np.cos(lat) * np.sin(lon)
        self.header['M3'] = d * np.sin(lat)

    def _update_from_cartesian(self, key, value):
        xd = self.header('M1')
        yd = self.header('M2')
        zd = self.header('M3')
        self.header['LON'] = np.arctan2(yd, xd)
        self.header['LAT'] = np.arctan2(np.sqrt(xd ** 2 + yd ** 2), zd)
        self.header['D'] = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)

class Object(fa.FitsArray):
    """
    A subclass of FitsArray with mandatory keywords defining an object cube
    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, header=None):
        if header is None:
            header = copy.copy(default_object_dict)
        # look for mandatory keywords
        for k in object_keys:
            if not header.has_key(k):
                header.update(k, default_object_dict[k])
        siddon.map_borders(header)

        obj = fa.FitsArray.__new__(subtype, shape, dtype=dtype,
                                   buffer=buffer, offset=offset,
                                   strides=strides, order=order,
                                   header=header)
        return obj

def circular_trajectory_data(**kargs):
    """
    Generate a circular trajectory of n images at a given radius

    Inputs
    ------
    radius : float
        radius of the trajectory
    dtype : data-type, optional (default np.float64)
        data type of the output array
    n_images : int (default 1)
        number of images
    min_lon : float (default 0.)
        first longitude value in radians
    max_lon : float (default 2 * np.pi)
        last longitude value in radians
    kargs :
        other keyword arguments are treated as keywords of
        the image header.

    Outputs
    -------

    data : InfoArray
        An empty InfoArray filled with appropriate metadata. The last axis
        is image index. The header elements are 1d arrays of length
        n_images.

    Exemple
    -------
    >>> data = circular_trajectory_data(**default_image_dict)
    >>> data.shape
    (1, 1, 1)
    """
    radius = kargs.pop('radius', 1.)
    dtype = kargs.pop('dtype', np.float64)
    n_images = kargs.pop('n_images', 1)
    min_lon = kargs.pop('min_lon', 0.)
    max_lon = kargs.pop('max_lon', 2 * np.pi)
    longitudes = np.linspace(min_lon, max_lon, n_images)
    images = []
    for i, lon in enumerate(longitudes):
        header = kargs.copy()
        shape = header['NAXIS1'], header['NAXIS2']
        images.append(Image(shape, header=dict(header), dtype=dtype))
        images[-1].update('LON', lon)
        images[-1].update('D', radius)
    data = fa.infoarrays2infoarray(images)
    # set values to zeros
    data[:] = 0.
    # compute rotation matrices
    siddon.full_rotation_matrix(data)
    return data

def object_from_header(header, **kwargs):
    """
    Generate an object from a given header
    """
    shape = header['NAXIS1'], header['NAXIS2'], header['NAXIS3']
    try:
        dtype = fa.bitpix[str(header['BITPIX'])]
    except(KeyError):
        dtype = np.float64
    dtype = header.pop('dtype', dtype)
    dtype = kwargs.pop('dtype', dtype)
    fill = kwargs.pop('fill', 0.)
    siddon.map_borders(header)
    obj = Object(shape, header=header, dtype=dtype, **kwargs)
    obj[:] = fill
    return obj

def spherical_object(**kargs):
    """
    Generate an object containing ones inside a sphere and zeros outside

    Inputs
    ------
    radius: float
        The radius of the sphere.
    NAXIS1: int
        The number of pixels along the first axis.
    NAXIS2: int
        The number of pixels along the second axis.
    NAXIS3: int
        The number of pixels along the third axis.
    dtype: data-type, optional (default: np.float64)

    Outputs
    -------
    obj: InfoArray
       A 3D infoarray with 1s inside a sphere and 0s outside

    """
    radius = kargs.pop('radius', 1.)
    obj = object_from_header(header)
    obj[:] = np.zeros(shape)
    x, y, z, = obj.axes()
    X, Y = np.meshgrid(x, y)
    for i in xrange(z.size):
        R = np.sqrt(X ** 2 + Y ** 2 + z[i] ** 2)
        obj[R < radius] = 1.
    return obj
