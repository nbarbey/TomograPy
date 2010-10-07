"""
Defines various tomography models and priors to be used with an optimizer.

- srt : Solar Rotational Tomography (with priors)
"""
import numpy as np
import lo
import siddon
from lo_wrapper import siddon_lo, siddon4d_lo
import secchi

def srt(data, cube, **kwargs):
    """
    Define Solar Rotational Tomography model with optional masking of
    data and map areas. Can also define priors.
    
    Parameters
    ----------
    data: InfoArray
        data cube
    cube: FitsArray
        map cube
    obj_rmin: float
        Object minimal radius. Areas below obj_rmin are masked out.
    obj_rmax: float
        Object maximal radius. Areas above obj_rmax are masked out.
    data_rmin: float
        Data minimal radius. Areas below data_rmin are masked out.
    data_rmax: float
        Data maximal radius. Areas above data_rmax are masked out.
    mask_negative: boolean
        If true, negative values in the data are masked out.

    Returns
    -------
    P : The projector with masking
    D : Smoothness priors

    """
    # Parse kwargs.
    obj_rmin = kwargs.get('obj_rmin', None)
    obj_rmax = kwargs.get('obj_rmax', None)
    data_rmin = kwargs.get('data_rmin', None)
    data_rmax = kwargs.get('data_rmax', None)
    mask_negative = kwargs.get('mask_negative', None)
    # Model : it is Solar rotational tomography, so obstacle="sun".
    P = siddon_lo(data.header, cube.header, obstacle="sun")
    D = [lo.diff(cube.shape, axis=i) for i in xrange(cube.ndim)]
    # Define masking.
    if obj_rmin is not None or obj_rmax is not None:
        Mo, obj_mask = mask_object(cube, kwargs)
        P = P * Mo.T
        D = [Di * Mo.T for Di in D]
    else:
        obj_mask = None
    if (data_rmin is not None or
        data_rmax is not None or
        mask_negative is not None):
        data_mask = secchi.define_data_mask(data,
                                            Rmin=data_rmin,
                                            Rmax=data_rmax,
                                            mask_negative=True)
        Md = lo.decimate(data_mask)
        P = Md * P
    else:
        data_mask = None
    return P, D, obj_mask, data_mask

def stsrt(data, cube, **kwargs):
    """
    Smooth Temporal Solar Rotational Tomography.
    Assumes data is sorted by observation time 'DATE_OBS'.
    """
    # Parse kwargs.
    obj_rmin = kwargs.get('obj_rmin', None)
    obj_rmax = kwargs.get('obj_rmax', None)
    data_rmin = kwargs.get('data_rmin', None)
    data_rmax = kwargs.get('data_rmax', None)
    mask_negative = kwargs.get('mask_negative', False)
    # define temporal groups
    times = [secchi.convert_time(t) for t in data.header['DATE_OBS']]
    ## if no interval is given separate every image
    dt_min = kwargs.get('dt_min', np.max(np.diff(times)) + 1)
    #groups = secchi.temporal_groups(data, dt_min)
    ind = secchi.temporal_groups_indexes(data, dt_min)
    n = len(ind)
    # 4d model
    cube_header = cube.header.copy()
    cube_header.update('NAXIS', 4)
    cube_header.update('NAXIS4', data.shape[-1])
    P = siddon4d_lo(data.header, cube_header, obstacle="sun")
    # define per group summation of maps
    # define new 4D cube
    cube4 = cube.reshape(cube.shape + (1,)).repeat(n, axis=-1)
    cube4.header.update('NAXIS', 4)
    cube4.header.update('NAXIS4', cube4.shape[3])
    cube4.header.update('CRVAL4', 0.)
    cube4.header.update('CDELT4', dt_min)
    S = group_sum(ind, cube, data)
    P = P * S.T
    # priors
    D = [lo.diff(cube4.shape, axis=i) for i in xrange(cube.ndim)]
    # mask object
    if obj_rmin is not None or obj_rmax is not None:
        Mo, obj_mask = mask_object(cube, kwargs)
        obj_mask = obj_mask.reshape(obj_mask.shape + (1,)).repeat(n, axis=-1)
        Mo = lo.decimate(obj_mask)
        P = P * Mo.T
        D = [Di * Mo.T for Di in D]
    else:
        obj_mask = None
    # mask data
    if (data_rmin is not None or
        data_rmax is not None or
        mask_negative is not None):
        data_mask = secchi.define_data_mask(data,
                                            Rmin=data_rmin,
                                            Rmax=data_rmax,
                                            mask_negative=True)
        Md = lo.decimate(data_mask)
        P = Md * P
    else:
        data_mask = None
    return P, D, obj_mask, data_mask, cube4

def mask_object(cube, kwargs):
    obj_rmin = kwargs.get('obj_rmin', None)
    obj_rmax = kwargs.get('obj_rmax', None)
    if obj_rmin is not None or obj_rmax is not None:
        obj_mask = secchi.define_map_mask(cube,
                                          Rmin=obj_rmin,
                                          Rmax=obj_rmax)
        Mo = lo.decimate(obj_mask)
    return Mo, obj_mask

def group_sum(ind, cube, data):
    from lo import ndsubclass
    # shapes
    shapein = cube.shape + (data.shape[-1],)
    shapeout = cube.shape + (len(ind),)
    # slicing indexes
    ind1 = ind
    ind2 = ind1[1:] + [None,]
    axis = -1
    def matvec(x):
        out = np.zeros(shapeout)
        for i, j, k in zip(ind1, ind2, np.arange(len(ind1))):
            out[..., k] += x[..., i:j].sum(axis=-1)
        return out
    def rmatvec(x):
        out = np.zeros(shapein)
        for i, j, k in zip(ind1, ind2, np.arange(len(ind1))):
            tmp_shape = out[..., i:j].shape
            out[..., i:j] = x[..., k].repeat(tmp_shape[-1]).reshape(tmp_shape)
        return out
    return lo.ndoperator(shapein, shapeout, matvec, rmatvec, dtype=np.float64)
