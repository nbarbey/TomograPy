"""
If lo package is present, define siddon lo wrapper
"""
import numpy as np
import lo
import fitsarray as fa
from siddon import dataarray_from_header, backprojector, projector
from siddon import backprojector4d, projector4d

class Siddon(lo.NDSOperator):
    def __init__(self, data_header, cube_header, **kwargs):
        self.data_header = data_header
        self.map_header = dict(cube_header)
        xin = fa.fitsarray_from_header(cube_header)
        xin.header = dict(xin.header)
        xin[:] = 0
        xout = dataarray_from_header(data_header)
        xout[:] = 0
        self.xin = xin
        self.xout = xout
        shapein = xin.shape
        shapeout = xout.shape
        def matvec(x):
            x = fa.InfoArray(data=x, header=dict(cube_header))
            y = xout
            y[:] = 0.
            projector(y, x, **kwargs)
            return y
        def rmatvec(x):
            x = fa.InfoArray(data=x, header=data_header)
            y = xin
            y[:] = 0.
            backprojector(x, y, **kwargs)
            return y
        lo.NDSOperator.__init__(self, shapein, shapeout, xin=xin, xout=xout,
                               matvec=matvec, rmatvec=rmatvec, dtype=xout.dtype)

class Siddon4d(lo.NDSOperator):
    def __init__(self, data_header, cube_header, ng=1, **kwargs):
        self.data_header = data_header
        self.map_header = cube_header
        xout = dataarray_from_header(data_header)
        xout[:] = 0
        xin = fa.fitsarray_from_header(cube_header)
        xin.header = dict(xin.header)
        xin[:] = 0
        self.xin = xin
        self.xout = xout
        def matvec(x):
            y = dataarray_from_header(data_header)
            y[:] = 0
            for i in xrange(ng):
                yi = y[..., i::ng]
                yi.header = y.header[i::ng]
                projector4d(yi, x, **kwargs)
            del yi
            return y
        def rmatvec(x):
            y = fa.fitsarray_from_header(cube_header)
            y[:] = 0
            for i in xrange(ng):
                xi = x[..., i::ng]
                xi.header = xi.header[i::ng]
                backprojector4d(xi, y, **kwargs)
            del xi
            return y
        lo.NDSOperator.__init__(self, xin=xin, xout=xout, matvec=matvec, rmatvec=rmatvec, dtype=xout.dtype)

def siddon_lo(data_header, cube_header, **kwargs):
    return Siddon(data_header, cube_header, **kwargs)

def siddon4d_lo(data_header, cube_header, **kwargs):
    return Siddon4d(data_header, cube_header, **kwargs)
