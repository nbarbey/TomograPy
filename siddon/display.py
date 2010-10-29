import numpy as np
from matplotlib import pyplot as plt

def data_movie(data, fig=None, **kwargs):
    """
    Display all images of a data cube as a movie.

    Arguments
    ---------
    data: ndarray
      A data cube. Third dimension should be the image index.
    fig: Figure instance (optional).
       A figure instance where the images will be displayed.
    kwargs: Keywor arguments.
       Other keyword arguments are passed to imshow.

    Returns
    -------
    Nothing.
    """
    if fig is None:
        fig = plt.figure()
    a = fig.gca()
    n = data.shape[-1]
    im0 = a.imshow(data[..., 0].T, **kwargs)
    plt.draw()
    for k in xrange(n):
        im0.set_data(data[..., k].T)
        plt.draw()

def sinogram(data, r, amin=-np.pi, amax=np.pi, n=None, fig=None, **kwargs):
    """
    Display a sinogram of the data set.

    Arguments
    ---------
    data: ndarray
      A data cube. Third dimension should be the image index.
    r: float
      The radius of the circle along which the sinogram is interpolated.
    rsun: boolean
      If true, r is the radius relative to the solar radius in the image.
    amin, amax: float (optional)
      Minimal and maximal angle. If not given amin=-pi and amax = pi.
    n: int (optional)
      Number of angular pixels. If not given, equals max of data.shape[0:2]
    fig: Figure instance (optional).
       A figure instance where the images will be displayed.
    kwargs: Keywor arguments.
       Other keyword arguments are passed to imshow.
    
    Return
    ------
    Returns the sinogram as a ndarray.
    """
    from scipy.ndimage import map_coordinates
    # handle kwargs
    if n is None:
        n = np.max(data.shape[0:2])
    # interpolat each frame
    a = np.linspace(amin, amax, n)
    sino = np.zeros((n, data.shape[-1]))
    for i in xrange(data.shape[-1]):
        # generate interpolation grid
        a0 = a - np.radians(data[i].header.get('SC_ROLL', 0.)[i])
        r0 = r * data[i].header.get('RSUN', 1.)[i] / 2
        x = r0 * np.cos(a0) + data[i].header.get('CRPIX1', 0.)[i]
        y = r0 * np.sin(a0) + data[i].header.get('CRPIX2', 0.)[i]
        sino[..., i] = map_coordinates(data[..., i], (x, y))
    # display sinogram
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    im0 = ax.imshow(sino, **kwargs)
    plt.draw()
    return sino

def display_object(obj):
    """
    Display an object map as an image by concatenating each slice
    along the third axis.
    """
    plt.figure()
    n = obj.shape[-1]
    div = _max_divider(n)
    shape = (obj.shape[0] * div, obj.shape[1] * n / div)
    tmp = np.zeros(shape)
    for k in xrange(n):
        i = k % div
        j = k / div
        i *= obj.shape[0]
        j *= obj.shape[1]
        tmp[i:i + obj.shape[0], j:j + obj.shape[1]] = obj[..., k]
    plt.imshow(tmp)
    plt.show()

def _max_divider(n):
    divmax = np.ceil(np.sqrt(n))
    divs = []
    for i in xrange(divmax):
        if np.remainder(n, i) == 0:
            divs.append(i)
    return np.max(divs)

# surface interpolations into the object

def extract_surface(obj, r, nlon=360., nlat=180., maxlat=np.pi,
                    pole="north", proj="equirectangular"):
    """
    Interpolate an equirectangular surface into an 3D object map.
    """
    from scipy.ndimage import map_coordinates
    if proj == "equirectangular":
        # 1d longitude and latitude
        lon = 2 * np.pi * np.linspace(0, 1, nlon)
        lat = np.pi * np.linspace(0, 1, nlat) - np.pi / 2.
        # 2d replicated coordinates
        Lon, Lat = np.meshgrid(lon, lat)
        # coordinate change
        x = r * np.cos(Lat) * np.cos(Lon)
        y = r * np.cos(Lat) * np.sin(Lon)
        z = r * np.sin(Lat)
        # rescale to pixel coordinates
        h = obj.header
        x = (x - h['CRVAL1']) / h['CDELT1'] + h['CRPIX1']
        y = (y - h['CRVAL2']) / h['CDELT2'] + h['CRPIX2']
        z = (z - h['CRVAL3']) / h['CDELT3'] + h['CRPIX3']
        out_map = map_coordinates(obj, (x, y, z))
    elif proj == "gnomon":
        # not yet implemented
        pass
    return out_map
