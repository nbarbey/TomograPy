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
# inversion
hyperparameters = 1. * numpy.ones(3)
savefile = '/tmp/siddon_ls.fits'
algo_type = Algorithms.SmoothQuadraticAlgorithm
algorithm = algo_type(siddon.direct_model, siddon.transpose_model,
                      hyperparameters, savefile=savefile)
algorithm.stop_criterion.n_max = 10
solution = algorithm(data, first_guess)
out_path = os.path.join(os.getenv('HOME'), 'data', 'siddon', 'output')
solution.to_fits(out_path + 'ls_solution.fits')
