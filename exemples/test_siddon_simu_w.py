#!/bin/env python
import numpy
import os
import copy
import time
import siddon
import Algorithms
# data 
data = siddon.Data((60, 256, 256))
data._generic_header(lon=numpy.pi, lat=-1., rol=-.08, )
data[:] = numpy.zeros(data.shape)
# cube
shape = 64. * numpy.ones(3)
crpix = shape / 2.
cdelt = 3. / shape
cube = siddon.Cube(shape, cdelt=cdelt, crpix=crpix)
cube[:] = numpy.zeros(cube.shape)
x, y, z = cube.axes()
X, Y, Z = siddon.meshgrid(x, y, z)
R = numpy.sqrt(X ** 2 + Y ** 2 + Z ** 2)
cube[(R > 1.) * (R < 1.4)] = 1.
# create data
t = time.time()
data = siddon.direct_model(cube, data)
print("projection time : " + str(time.time() - t))
# get first map
t = time.time()
first_guess = siddon.transpose_model(cube, data)
print("backprojection time : " + str(time.time() - t))
# weights
weights = siddon.transpose_model(cube, data.ones())
coverage = siddon.transpose_model(first_guess, data.ones())
weights = 1 / numpy.sqrt(coverage)
weights[numpy.isinf(weights)] = 0
weights[:] = 1.

# inversion
direct_model = siddon.DirectModel(weights)
transpose_model = siddon.TransposeModel(weights)
hyperparameters = 1. * numpy.ones(3)
savefile = '/tmp/siddon_sim_ls.fits'
algo_type = Algorithms.SmoothQuadraticAlgorithm
algorithm = algo_type(direct_model, transpose_model,
                      hyperparameters, savefile=savefile)
solution = algorithm(data, first_guess)
solution /= weights
solution[R < 1.05] = 0
#save
out_path = os.path.join(os.getenv('HOME'), 'data', 'siddon', 'output')
solution.to_fits(out_path + 'ls_solution.fits')
