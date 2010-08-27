#!/bin/env python
import numpy
import os
import copy
import time
import siddon
import Algorithms
# data 
path = os.path.join(os.getenv('HOME'), 'data', '171dec08')
obsrvtry = 'STEREO_A'
time_window = ['2008-12-01T00:00:00.000', '2008-12-30T00:00:00.000']
time_step = 4 * 3600. # one image every time_step seconds
data = siddon.read_siddon_data(path, obsrvtry=obsrvtry, 
                               time_window=time_window, time_step=time_step)
data = data.rebin(4)
# cube
shape = 128 * numpy.ones(3)
crpix = shape / 2.
d = 3 * numpy.ones(3)
cdelt = d / shape
first_guess = siddon.Cube(shape, cdelt=cdelt, crpix=crpix)
first_guess[:] = numpy.zeros(first_guess.shape)
t = time.time()
first_guess = siddon.transpose_model(first_guess, data)
print("backprojection time : " + str(time.time() - t))
coverage = siddon.transpose_model(first_guess, data.ones())
weights = 1 / numpy.sqrt(coverage)
weights[numpy.isinf(weights)] = 0

# inversion
hyperparameters = 0 * numpy.ones(3)
savefile = '/tmp/siddon_lsw'
algo_type = Algorithms.SmoothQuadraticAlgorithm
direct_model = siddon.DirectModel(weights)
transpose_model = siddon.TransposeModel(weights)
algorithm = algo_type(direct_model, transpose_model, hyperparameters, 
                      savefile=savefile)
solution = algorithm(data, first_guess)
out_path = os.path.join(os.getenv('HOME'), 'data', 'siddon', 'output')
solution.to_fits(out_path + 'ls_weights_solution.fits')
