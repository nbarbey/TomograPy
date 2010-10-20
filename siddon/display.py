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
    im0 = a.imshow(data[..., 0], **kwargs)
    plt.draw()
    for k in xrange(n):
        im0.set_data(data[..., k])
        plt.draw()

def sinogram(data, r, amin=None, amax=None, n=None, fig=None, **kwargs):
    """
    Display a sinogram of the data set.

    Arguments
    ---------
    data: ndarray
      A data cube. Third dimension should be the image index.
    r: float
      The radius of the circle along which the sinogram is interpolated.
    amin, amax: float (optional)
      Minimal and maximal angle. If not given amin=0 and amax = pi.
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
    if amin is None:
        amin = 0.
    if amax is None:
        amax = np.pi
    if n is None:
        n = np.max(data.shape[0:2])
    # interpolat each frame
    sino = np.zeros((n, data.shape[-1]))
    for i in xrange(data.shape[-1]):
        # generate interpolation grid
        a = np.linspace(amin, amax, n)
        x = r * np.cos(a)
        y = r * np.sin(a)
        x += data[i].header['CRPIX1'][i]
        y += data[i].header['CRPIX2'][i]
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
