"""
If lo package is present, define siddon lo wrapper
"""
import lo
import fitsarray as fa
from siddon import dataarray_from_header, backprojector, projector

def siddon_lo(data_header, cube_header, obstacle=None):
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
        projector(y, x, obstacle=obstacle)
        return y
    def rmatvec(x):
        y = fa.fitsarray_from_header(cube_header)
        y[:] = 0
        backprojector(x, y, obstacle=obstacle)
        return y
    return lo.ndsubclass(cube, data, matvec=matvec, rmatvec=rmatvec, dtype=data.dtype)

def siddon4d_lo(data_header, cube_header, obstacle=None):
    """
    A linear operator performing projection and backprojection using
    Siddon 4-dimensional variation.
    """
    data = dataarray_from_header(data_header)
    data[:] = 0
    cube = fa.fitsarray_from_header(cube_header)
    cube[:] = 0
    def matvec(x):
        y = dataarray_from_header(data_header)
        y[:] = 0
        projector4d(y, x, obstacle=obstacle)
        return y
    def rmatvec(x):
        y = fa.fitsarray_from_header(cube_header)
        y[:] = 0
        backprojector4d(x, y, obstacle=obstacle)
        return y
    return lo.ndsubclass(cube, data, matvec=matvec, rmatvec=rmatvec, dtype=data.dtype)
