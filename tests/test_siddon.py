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

# check that projection does not fail
def check_projector(im_h, obj_h):
    obj = siddon.simu.object_from_header(obj_h)
    data = siddon.simu.circular_trajectory_data(**im_h)
    if data.dtype == obj.dtype:
        siddon.projector(data, obj)

def test_projector():
    for im_h in image_headers:
        for obj_h in object_headers:
            yield check_projector, im_h, obj_h


# check per image projection vs full projection
# (for exemple openmp issues could be catched here
def check_full_versus_image(im_h, obj_h):
    obj = siddon.simu.object_from_header(obj_h)
    data = siddon.simu.circular_trajectory_data(**im_h)
    data2 = siddon.simu.circular_trajectory_data(**im_h)
    if data.dtype == obj.dtype:
        siddon.projector(data, obj)
        for t in xrange(data2.shape[-1]):
            siddon.conic_image_projector(data2, obj, t)
        assert_array_almost_equal(data, data2)

def test_full_versus_image():
    for im_h in image_headers:
        for obj_h in object_headers:
            yield check_full_versus_image, im_h, obj_h

# check full unit vector
def check_full_unit_vector(im_h):
    data = siddon.simu.circular_trajectory_data(**im_h)
    u = siddon.siddon.C_full_unit_vector(data)
    upy = siddon.siddon.full_unit_vector(data)
    assert_array_equal(u, upy)

def test_full_unit_vector():
    for im_h in image_headers:
        yield check_full_unit_vector, im_h

# check intersection parameters
def check_full_intersection_parameters(im_h, obj_h):
    data = siddon.simu.circular_trajectory_data(**im_h)
    obj = siddon.fa.fitsarray_from_header(obj_h)
    if data.dtype == obj.dtype:
        u = siddon.siddon.C_full_unit_vector(data)
        a1py, anpy = siddon.siddon.full_intersection_parameters(data, obj, u)
        a1, an = siddon.siddon.C_full_intersection_parameters(data, obj, u)

        assert_array_equal(a1, a1py)
        assert_array_equal(an, anpy)

def test_full_intersection_parameters():
    for im_h in image_headers:
        for obj_h in object_headers:
            yield check_full_intersection_parameters, im_h, obj_h

# check that obj.header['PSHAPE'] is returned for ray parallel to axes

def test_parallel_ray():
    data = siddon.simu.circular_trajectory_data(**image_headers[0])
    data.header[0]['CRPIX1'] = 1.
    data.header[0]['CRPIX2'] = 1.
    for i, lon, lat in ((0, 0, 0), (1, np.pi / 2, 0), (2, np.pi / 2, np.pi / 2)):
        for obj_h in object_headers64:
            obj = siddon.fa.fitsarray_from_header(obj_h)
            obj[:] = 0.
            if data.dtype == obj.dtype:
                data[:] = 1.
                siddon.siddon.backprojector(data, obj)
                assert_equal(np.sum(obj), obj.header['PSHAPE' + str(i + 1)])

# test that the sum of a bpj in a big cube equal the projection in a
# cube of 1 pix of the same shape

def test_sum_bpj():
    data = siddon.simu.circular_trajectory_data(**image_headers[0])
    data[:] = 1.
    obj_h1 = object_headers64[-1]
    obj_h2 = object_headers64[0]
    obj1 = siddon.fa.fitsarray_from_header(obj_h1)
    obj2 = siddon.fa.fitsarray_from_header(obj_h2)
    obj1[:] = 0.
    obj2[:] = 0.
    siddon.siddon.backprojector(data, obj1)
    siddon.siddon.backprojector(data, obj2)
    assert_equal(np.sum(obj1), obj2)
