#!/usr/bin/env python

import nose
from numpy.testing import *
import numpy as np
import siddon

# metadata
minimal_object_header =  {'SIMPLE':True,'BITPIX':-64,
                          'NAXIS1':1, 'NAXIS2':1, 'NAXIS3':1,
                          'CRPIX1':.5, 'CRPIX2':.5, 'CRPIX3':.5,
                          'CDELT1':1., 'CDELT2':1., 'CDELT3':1.,
                          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}

minimal_image_header = {'n_images':1,
                        'SIMPLE':True, 'BITPIX':-64,
                        'NAXIS1':1, 'NAXIS2':1,
                        'CRPIX1':0., 'CRPIX2':0.,
                        'CDELT1':1., 'CDELT2':1.,
                        'CRVAL1':0., 'CRVAL2':0.,
                        }
obj0 = siddon.simu.object_from_header(minimal_object_header)
obj0[:] = 1.

minimal_image_header['radius'] = 1e6
minimal_image_header['max_lon'] = np.pi
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
