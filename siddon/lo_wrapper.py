"""
If lo package is present, define siddon lo wrapper
"""
import lo
import fitsarray as fa
from siddon import dataarray_from_header, backprojector, projector

def siddon_lo(data_header, cube_header):
    """
    A linear operator performing projection and backprojection using
    Siddon.
    """
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

def siddon_sun_lo(data_header, cube_header):
    """
    A linear operator performing projection and backprojection using
    Siddon.
    Rays are blocked at a centered sphere of radius one (as it is the
    case in solar tomography).
    """
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
