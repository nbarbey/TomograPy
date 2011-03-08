#!/usr/bin/env python
"""
Perform (back)projections with various number of images and cores to
test performances and scalability of the code.
"""
import time
import multiprocessing as mp
import numpy as np
import tomograpy
from test_cases import *

nthread_max = mp.cpu_count()
d = 200

image_shapes = [128, 256, 512, 1024]
cube_shapes = [128, 256, 512, 1024]
cube_shape0 = 128
data_shape0 = 512
n_images = 64

def test_cores():
    obj = tomograpy.centered_cubic_map(3, cube_shape0)
    data = tomograpy.centered_stack(tomograpy.fov(obj, d), data_shape0, n_images=n_images)
    # projection
    pj_times = np.empty(nthread_max + 1)
    pj_times[0] = time.time()
    for nt in xrange(nthread_max):
        tomograpy.projector(data, obj, nthread=nt + 1)
        pj_times[nt + 1] = time.time()
    pj_times = pj_times[1:] - pj_times[:-1]
    # backprojection
    bpj_times = np.empty(nthread_max + 1)
    bpj_times[0] = time.time()
    for nt in xrange(nthread_max):
        tomograpy.backprojector(data, obj, nthread=nt + 1)
        bpj_times[nt + 1] = time.time()
    bpj_times = bpj_times[1:] - bpj_times[:-1]
    # pretty print
    text = ''
    text += 'Cores'
    text += ''.join([' & ' + str(i + 1)  for i in xrange(nthread_max)])
    text += ' \\\\ \n' + 'Projection (s)'
    text += ''.join([' & %2.2f' % pjt for pjt in pj_times])
    text += ' \\\\ \n' + 'Backprojection (s)'
    text += ''.join([' & %2.2f' % bpjt for bpjt in bpj_times])
    text += ' \\\\ \n'
    print text

def test_n_images():
    NotImplemented

def test_image_shape():
    obj = tomograpy.centered_cubic_map(3, cube_shape0)
    # projection
    pj_times = np.empty(len(image_shapes) + 1)
    pj_times[0] = time.time()
    for i, s in enumerate(image_shapes):
        data = tomograpy.centered_stack(tomograpy.fov(obj, d), s, n_images=n_images)
        tomograpy.projector(data, obj)
        pj_times[i + 1] = time.time()
    pj_times = pj_times[1:] - pj_times[:-1]
    # backprojection
    bpj_times = np.empty(len(image_shapes) + 1)
    bpj_times[0] = time.time()
    for i, s in enumerate(image_shapes):
        data = tomograpy.centered_stack(tomograpy.fov(obj, d), s, n_images=n_images)
        tomograpy.backprojector(data, obj)
        bpj_times[i + 1] = time.time()
    bpj_times = bpj_times[1:] - bpj_times[:-1]
    # pretty print
    text = ''
    text += 'Image shape'
    text += ''.join([' & ' + str(s) + " $\\times$ " + str(s)  for s in image_shapes])
    text += ' \\\\ \n' + 'Projection (s)'
    text += ''.join([' & %2.2f' % pjt for pjt in pj_times])
    text += ' \\\\ \n' + 'Backprojection (s)'
    text += ''.join([' & %2.2f' % bpjt for bpjt in bpj_times])
    text += ' \\\\ \n'
    print text

def test_map_shape():
    obj = tomograpy.centered_cubic_map(3, cube_shape0)
    data = tomograpy.centered_stack(tomograpy.fov(obj, d), data_shape0, n_images=n_images)
    # projection
    pj_times = np.empty(len(cube_shapes) + 1)
    pj_times[0] = time.time()
    for i, s in enumerate(cube_shapes):
        obj = tomograpy.centered_cubic_map(3, s)
        tomograpy.projector(data, obj)
        pj_times[i + 1] = time.time()
    pj_times = pj_times[1:] - pj_times[:-1]
    # backprojection
    bpj_times = np.empty(len(cube_shapes) + 1)
    bpj_times[0] = time.time()
    for i, s in enumerate(cube_shapes):
        obj = tomograpy.centered_cubic_map(3, s)
        tomograpy.backprojector(data, obj)
        bpj_times[i + 1] = time.time()
    bpj_times = bpj_times[1:] - bpj_times[:-1]
    # pretty print
    text = ''
    text += 'Cube shape'
    text += ''.join([' & ' + str(s) + "$^3$"  for s in cube_shapes])
    text += ' \\\\ \n' + 'Projection (s)'
    text += ''.join([' & %2.2f' % pjt for pjt in pj_times])
    text += ' \\\\ \n' + 'Backprojection (s)'
    text += ''.join([' & %2.2f' % bpjt for bpjt in bpj_times])
    text += ' \\\\ \n'
    print text

if __name__ == "__main__":
    test_cores()
    test_n_images()
    test_image_shape()
    test_map_shape()
