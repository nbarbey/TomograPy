#!/usr/bin/env python

import nose
from numpy.testing import *
import numpy as np
import siddon
import fitsarray as fa
import lo

from test_cases import *

models = [siddon.models.srt, siddon.models.stsrt, ]

def check_model(model, im_h, obj_h):
    obj = siddon.simu.object_from_header(obj_h)
    obj = siddon.fa.InfoArray(data=obj, header=dict(obj.header))
    obj[:] = 1.
    im_h['n_images'] = 100
    im_h['max_lon'] = 3 * np.pi
    data = siddon.simu.circular_trajectory_data(**im_h)
    if obj.dtype == data.dtype:
        P, D, obj_mask, data_mask = model(data, obj, obj_rmin=1.)
        w = (P.T * np.ones(data.size)).reshape(obj_mask.shape)
        is_seen = (w != 0)
        new_obj = fa.FitsArray(obj_mask.shape)
        new_obj = 1.
        new_obj *= (1 - obj_mask)
        data[:] = (P * new_obj.ravel()).reshape(data.shape)
        hypers = new_obj.ndim * (1e-10, )
        sol = lo.acg(P, data.ravel(), D, hypers=hypers, tol=1e-10)
        sol = fa.asfitsarray(sol.reshape(obj_mask.shape), header=obj.header)
        assert_almost_equal(sol[is_seen], new_obj[is_seen], decimal=1)

def test_models():
    for model in models:
        yield check_model, model, image_headers64[1], object_headers64[1]

if __name__ == "__main__":
    nose.run(argv=['', __file__])
