#!/bin/env python
import numpy
import os
import copy
import time
import siddon
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
cube = siddon.Cube(shape, cdelt=cdelt, crpix=crpix)
cube[:] = numpy.ones(cube.shape)
data[:] = numpy.zeros(data.shape)
t = time.time()
siddon.projector(data, cube)
print("backprojection time : " + str(time.time() - t))
