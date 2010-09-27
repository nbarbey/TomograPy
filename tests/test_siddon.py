import nose
from numpy.testing import *
import numpy as np
import siddon

def test_simu():
    import siddon.simu as simu
    
    im = simu.fa.infoarrays2infoarray([simu.Image((1, 1)),])
    obj = simu.Object((1, 1, 1))
    P = siddon.siddon_lo(im.header, obj.header, obstacle=None)
    assert_almost_equal(P.todense()[0], 0.39369162)

if __name__ == "__main__":
    nose.run(argv=['', __file__])
