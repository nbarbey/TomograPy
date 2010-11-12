#!/usr/bin/env python
import numpy as np
import os
import copy
import time
import siddon
import fitsarray as fa
# data 
path = os.path.join(os.getenv('HOME'), 'data', 'siddon', '171dec08')
obsrvtry = 'STEREO_A'
time_window = ['2008-12-01T00:00:00.000', '2008-12-15T00:00:00.000']
time_step = 32 * 3600. # one image every time_step seconds
data = siddon.secchi.read_data(path, bin_factor=64,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
# cube
shape = 3 * (16,)
header = {'CRPIX1':shape[0] / 2., 'CRPIX2':shape[1] / 2., 'CRPIX3':shape[2] / 2.,
          'CDELT1':3. / shape[0], 'CDELT2':3. / shape[1], 'CDELT3':3. / shape[2],
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
cube = fa.zeros(shape, header=header)
t = time.time()
cube = siddon.backprojector(data, cube, obstacle="sun")
print("backprojection time : " + str(time.time() - t))
