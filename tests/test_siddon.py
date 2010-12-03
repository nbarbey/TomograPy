#!/usr/bin/env python

import nose
from numpy.testing import *
import numpy as np
import siddon
import fitsarray as fa

from test_cases import *

def check_simu_shape(h):
    obj0 = siddon.simu.object_from_header(h)
    assert_equal(obj0.shape, (h['NAXIS1'], h['NAXIS2'], h['NAXIS3']))

def test_object_from_header():
    for h in object_headers:
        yield check_simu_shape, h

def check_simu_dtype(h):
    obj0 = siddon.simu.object_from_header(h)
    dtype = fa.bitpix[str(h['BITPIX'])]
    assert_equal(obj0.dtype.name, dtype)

def test_object_from_header_dtype():
    for h in object_headers:
        yield check_simu_dtype, h

def check_projector(im_h, obj_h):
    obj = siddon.simu.object_from_header(obj_h)
    data = siddon.simu.circular_trajectory_data(**im_h)
    if data.dtype == obj.dtype:
        siddon.projector(data, obj)

def test_projector():
    for im_h in image_headers:
        for obj_h in object_headers:
            yield check_projector, im_h, obj_h

obj0 = siddon.simu.object_from_header(minimal_object_header)
obj0[:] = 1.

data0 = siddon.simu.circular_trajectory_data(**minimal_image_header)
data0[:] = 1.

# class to generate test
class TestSiddon(object):
    def __init__(self, func, data0, obj0):
        self.func = func
        self.obj0 = obj0
        self.data0 = data0
    def __call__(self, data, obj, **kwargs):
        self.func(data, obj, self.kwargs)
        assert_almost_equal(obj, obj0)
        assert_almost_equal(data, data0)

test_projector_minimal = TestSiddon(siddon.projector, data0, obj0)

def test_simu():
    import siddon.simu as simu
    
    im = simu.fa.infoarrays2infoarray([simu.Image((1, 1)),])
    obj = simu.Object((1, 1, 1))
    im[:] = 0.
    obj[:] = 1.
    im2 = siddon.projector(im.copy(), obj)
    assert_almost_equal(im2[0], 0.39369162)
    im[:] = 1.
    obj[:] = 0.
    obj2 = siddon.backprojector(im, obj.copy())
    assert_almost_equal(obj2[0], 0.39369162)

if __name__ == "__main__":
    nose.run(argv=['', __file__])
