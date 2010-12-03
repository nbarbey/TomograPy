#!/usr/bin/env python

import nose
from numpy.testing import *
import numpy as np
import siddon
import fitsarray as fa
import lo

from test_cases import *

models = [siddon.models.srt, siddon.models.stsrt, siddon.models.thomson]

def check_model(model, im_h, obj_h):
    obj = siddon.simu.object_from_header(obj_h)
    data = siddon.simu.circular_trajectory_data(**im_h)
    if obj.dtype == data.dtype:
        P, D, obj_mask, data_mask = model(data, obj)
        data = P * obj.ravel()
        hypers = obj.ndim * (1., )
        sol = lo.acg(P, data.ravel(), D, hypers=hypers)
        sol = fa.asfitsarray(sol.reshape(obj.shape))
        assert_almost_equal(obj, sol)

def test_models():
    for model in models:
        yield check_model, model, small_image_header, small_object_header

if __name__ == "__main__":
    nose.run(argv=['', __file__])
