"""
Defines various tomography models and priors to be used with an optimizer.

- srt : Solar Rotational Tomography (with priors)
"""
import numpy as np
import lo
import siddon
from lo_wrapper import siddon_lo
import secchi

def srt(data, cube, **kargs):
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
    # Parse kargs.
    obj_rmin = kargs.pop('obj_rmin', None)
    obj_rmax = kargs.pop('obj_rmax', None)
    data_rmin = kargs.pop('data_rmin', None)
    data_rmax = kargs.pop('data_rmax', None)
    mask_negative = kargs.pop('mask_negative', None)
    # Model : it is Solar rotational tomography, so obstacle="sun".
    P = siddon_lo(data.header, cube.header, obstacle="sun")
    D = [lo.diff(cube.shape, axis=i) for i in xrange(cube.ndim)]
    # Define masking.
    if obj_rmin is not None or obj_rmax is not None:
        obj_mask = secchi.define_map_mask(cube,
                                          Rmin=obj_rmin,
                                          Rmax=obj_rmax)
        Mo = lo.decimate(obj_mask)
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


def stsrt(data, cube, **kargs):
    """
    Smooth Temporal Solar Rotational Tomography.
    Assumes data is sorted by observation time 'DATE_OBS'.
    """
    times = [secchi.convert_time(t) for t in data.header['DATE_OBS']]
    # if not interval is given separate every image
    dt_min = kargs.pop('dt_min', np.max(np.diff(times)) + 1)
    groups = secchi.temporal_groups(data, dt_min)
    # number of groups
    n = len(groups)
    # define the 4D projector
    Pl = []
    dmask_list = []
    for d in groups:
        Pi, Di, obj_mask_i, data_mask_i = srt(d, cube, **kargs)
        Pl.append(Pi)
        dmask_list.append(data_mask_i)
    # define new 4D cube
    cube = cube.reshape(cube.shape + (1,)).repeat(n, axis=-1)
    obj_mask = obj_mask_i.reshape(obj_mask_i.shape + (1,)).repeat(n, axis=-1)
    data_mask = np.concatenate(dmask_list, axis=-1)
    P = lo.block_diagonal(Pl)
    D = [lo.diff(cube.shape, axis=i) for i in xrange(4)]
    # mask for priors
    Mo = lo.decimate(obj_mask)
    D = [Di * Mo.T for Di in D]
    return P, D, obj_mask, data_mask, cube
