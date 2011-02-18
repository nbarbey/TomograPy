import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

def data_movie(data, fig=None, pause=None, **kwargs):
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
    dmin, dmax = data.min(), data.max()
    n = data.shape[-1]
    im0 = a.imshow(data[..., 0].T, origin="lower", **kwargs)
    plt.draw()
    for k in xrange(n):
        im0.set_data(data[..., k].T)
        plt.title("Image " + str(k))
        plt.clim([dmin, dmax])
        plt.draw()
        if pause is not None:
            time.sleep(pause)

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
    from solar import compute_rsun
    # handle kwargs
    if n is None:
        n = np.max(data.shape[0:2])
    # interpolat each frame
    a = np.linspace(amin, amax, n)
    sino = np.zeros((n, data.shape[-1]))
    rsun = compute_rsun(data)
    for i in xrange(data.shape[-1]):
        # generate interpolation grid
        a0 = a - np.radians(data[i].header.get('SC_ROLL', 0.)[i])
        r0 = r * rsun[i] / 2
        x = r0 * np.cos(a0) + data[i].header.get('CRPIX1', 0.)[i]
        y = r0 * np.sin(a0) + data[i].header.get('CRPIX2', 0.)[i]
        sino[..., i] = map_coordinates(data[..., i], (x, y))
    # display sinogram
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    im0 = ax.imshow(sino, origin="lower", **kwargs)
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
    plt.imshow(tmp, origin="lower")
    plt.draw()

def _max_divider(n):
    divmax = int(np.ceil(np.sqrt(n)))
    divs = []
    for i in xrange(divmax):
        if np.remainder(n, i) == 0:
            divs.append(i)
    return np.max(divs)

# surface interpolations into the object
def equirectangular(obj, r=None, nlon=360., nlat=180., maxlat=np.pi, **kwargs):
    """
    Equirectangular projection.
    """
    # 1d longitude and latitude
    lon = 2 * np.pi * np.linspace(0, 1, nlon)
    lat = np.pi * np.linspace(0, 1, nlat) - np.pi / 2.
    # 2d replicated coordinates
    Lon, Lat = np.meshgrid(lon, lat)
    # coordinate change
    x, y, z = sphe2cart(r, Lon, Lat)
    # rescale to pixel coordinates
    x, y, z = phy2pix(obj.header, (x, y, z))
    # interpolate and return
    return map_coordinates(obj, (x, y, z))

def gnomonic(obj, r=None, nlon=360., nlat=180., maxlat=3 * np.pi / 4.,
             pole="north", **kwargs):
    """
    Gnomonic projection.
    """
    # select pole
    convert_pole = {"north":1, "south":-1}
    sign = convert_pole[pole]
    # generate 1d coordinates
    x = maxlat * np.linspace(-.5, .5, obj.shape[0])
    y = maxlat * np.linspace(-.5, .5, obj.shape[1])
    # replicate in 2d.
    X, Y = np.meshgrid(x, y)
    lon = np.arctan2(Y, X)
    lat = sign * (np.pi / 2. - np.sqrt(X ** 2 + Y ** 2))
    # convert in cartesian coordinates
    x, y, z = sphe2cart(r, lon, lat)
    # convert in pixel coordinates
    x, y, z = phy2pix(obj.header, (x, y, z))
    # interpolate and return
    return map_coordinates(obj, (x, y, z))

def orthographic(obj, r=None, **kwargs):
    """
    Orthographic projection.
    """
    # generate 1d coordinates
    x = np.linspace(-.5, .5, obj.shape[0])
    y = np.linspace(-.5, .5, obj.shape[1])
    # replicate in 2d.
    X, Y = np.meshgrid(x, y)
    Z2 = r ** 2 - X ** 2 - Y ** 2
    # if negative values drop them
    #goods = np.where(Z2 > 0.)
    #X = X[goods]
    #Y = Y[goods]
    #Z = np.sqrt(Z2[goods])
    Z = np.sqrt(Z2)
    # convert in pixel coordinates
    X, Y, Z = phy2pix(obj.header, (X, Y, Z))
    # interpolate and return
    return map_coordinates(obj, (X, Y, Z))

# coordinates transformations
def sphe2cart(r, lon, lat):
    """Spherical to cartesian coordinates.
    """
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def phy2pix(header, coords):
    """Transform physical coordinates into pixel coordinates.

    Arguments
    ---------
    header: dict or pyfits.Header
       A header with CRVAL, CDELT and CRPIX keywords.
    coords: list of arrays
       The coordinates stored in an iterable.
       
    Returns
    -------
    out_coords: list of ndarrays
      The coordinates in pixels.
    """
    h = header
    out_coords = list()
    for i, u in enumerate(coords):
        si = str(i + 1)
        out_coords.append((u - h['CRVAL' + si]) / h['CDELT' + si] + h['CRPIX' + si])
    return out_coords

# define the dict of map projections and corresponding name string
_map_projections = [equirectangular, gnomonic, orthographic]

def _define_map_projections():
    out = dict()
    for proj in _map_projections:
        out[proj.__name__] = proj
    return out

map_projections = _define_map_projections()
del _map_projections

# generic surface extraction and display
def display_surface(obj, proj, title='', xlabel='', ylabel='',
                    imshow_kwargs={}, **kwargs):
    """
    Calls extract surface and imshow.
    """
    my_proj = extract_surface(obj, proj, **kwargs)
    plt.imshow(my_proj, origin="lower", **imshow_kwargs)
    plt.draw()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return my_proj

def extract_surface(obj, proj, **kwargs):
    """
    Interpolate a surface into an 3D object map according to a projection.

    Arguments
    ---------
    obj: FitsArray.
      A 3d fits array containing the map to interpolate.
    proj: see display.map_projections
      The name of the projection to perform.
    kwargs: keyword arguments
      The arguments which need to be passed to the projections routines.
    Returns
    -------
    A ndarrray containing the interpolated map.
    """
    if isinstance(proj, str):
        proj = map_projections[proj]
    if proj in map_projections.values():
        return proj(obj, **kwargs)
    else:
        raise ValueError("projection %s not implemented." % proj )
