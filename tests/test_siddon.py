#!/usr/bin/env python

import nose
from numpy.testing import *
import numpy as np
import siddon

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
