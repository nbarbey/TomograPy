#!/usr/bin/env python
import os
import time
import numpy as np
import lo
import siddon
from siddon.solar import read_data
# data
data = read_data(os.path.join(os.getenv('HOME'), 'data', 'siddon', '171dec08'),
                 bin_factor=4.,
                 time_window=['2008-12-01T00:00:00.000',
                              '2008-12-15T00:00:00.000'],
                 time_step=8 * 3600.
                 )
data = siddon.solar.sort_data_array(data)
# scale A and B images
# the ratio of sensitivity between EUVI A and B
calibration_ba = {171:0.902023, 195:0.974536, 284:0.958269, 304:1.05954}
for i in xrange(data.shape[-1]):
    if data.header[i]['OBSRVTRY'] == 'STEREO_B':
        data[..., i] /= calibration_ba[data.header[i]['WAVELNTH']]

    # make sure it is 64 bits data
    data.header[i]['BITPIX'] = -64

# cube
cube = siddon.siddon.centered_cubic_map(3, 64, fill=0.)
# model
kwargs = {'obj_rmin':1., 'obj_rmax':1.5, 'data_rmin':1., 'data_rmax':1.3,
          'mask_negative':False, 'mask_nan':True}
P, D, obj_mask, data_mask = siddon.models.srt(data, cube, **kwargs)
# apply mask to data
data *= (1 - data_mask)
data[np.isnan(data)] = 0.
# hyperparameters
hypers = cube.ndim * (1e0, )
# inversion
# expected time
b = data.ravel()
b[np.isnan(b)] = 0.
t = time.time()
bpj = P.T * b
print((time.time() - t) * 4 * 100 )
# real time
t = time.time()
sol = lo.acg(P, b, D, hypers, maxiter=100)
print(time.time() - t)
# reshape result
fsol = siddon.fa.asfitsarray(sol.reshape(cube.shape), header=cube.header)
siddon_path = os.path.join(os.getenv('HOME'), 'data', 'siddon')
fsol.tofits(os.path.join(siddon_path, "output", "test_siddon_secchi_mask.fits"))
