#!/usr/bin/env python
import numpy as np
import os
import copy
import time
import tomograpy
import fitsarray as fa
# data 
path = os.path.join(os.getenv('HOME'), 'data', 'tomograpy., '171dec08')
obsrvtry = 'STEREO_A'
time_window = ['2008-12-01T00:00:00.000', '2008-12-15T00:00:00.000']
time_step = 8 * 3600. # one image every time_step seconds
data = tomograpy.solar.read_data(path, bin_factor=8,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
# map
cube = tomograpy.centered_cubic_map(3, 128, fill=0.)
t = time.time()
cube = tomograpy.backprojector(data, cube, obstacle="sun")
print("backprojection time : " + str(time.time() - t))
