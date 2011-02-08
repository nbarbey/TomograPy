#!/usr/bin/env python
"""
Perform (back)projections with various number of images and cores to
test performances and scalability of the code.
"""
import time
import multiprocessing as mp
import numpy as np
import siddon
from test_cases import *

nthread_max = mp.cpu_count()
d = 200

def test_cores():
    obj = siddon.centered_cubic_map(3, 256)
    data = siddon.centered_stack(siddon.fov(obj, d), 128, n_images=64)
    # projection
    pj_times = np.empty(nthread_max + 1)
    pj_times[0] = time.time()
    for nt in xrange(nthread_max):
        siddon.projector(data, obj, nthread=nt + 1)
        pj_times[nt + 1] = time.time()
    pj_times = pj_times[1:] - pj_times[:-1]
    # backprojection
    bpj_times = np.empty(nthread_max + 1)
    bpj_times[0] = time.time()
    for nt in xrange(nthread_max):
        siddon.backprojector(data, obj, nthread=nt + 1)
        bpj_times[nt + 1] = time.time()
    bpj_times = bpj_times[1:] - bpj_times[:-1]
    # pretty print
    text = ''
    text += 'Cores'
    text += ''.join([' & ' + str(i + 1)  for i in xrange(nthread_max)])
    text += ' \\\\ \n' + 'Projection (s)'
    text += ''.join([' & ' + str(pjt) for pjt in pj_times])
    text += ' \\\\ \n' + 'Backprojection (s)'
    text += ''.join([' & ' + str(bpjt) for bpjt in bpj_times])
    text += ' \\\\ \n'
    print text

def test_n_images():
    NotImplemented

def test_image_shape():
    NotImplemented

def test_map_shape():
    NotImplemented

if __name__ == "__main__":
    test_cores()
    test_n_images()
    test_image_shape()
    test_map_shape
