#!/usr/bin/env python
import os
import time
import lo
import siddon
from siddon.secchi import read_data
#
siddon_path = os.path.join(os.getenv('HOME'), 'data', 'siddon')
# data
obsrvtry = ('STEREO_A', 'STEREO_B')
data = siddon.secchi.concatenate(
    [read_data(os.path.join(siddon_path, '171dec08'), 
               bin_factor=4,
               obsrvtry=obs,
               time_window=['2008-12-01T00:00:00.000', 
                            '2008-12-15T00:00:00.000'],
               time_step=4 * 3600.
               )
     for obs in obsrvtry])
data = siddon.secchi.sort_data_array(data)
# scale A and B images
# the ratio of sensitivity between EUVI A and B
calibration_ba = {171:0.902023, 195:0.974536, 284:0.958269, 304:1.05954}
for i in xrange(data.shape[-1]):
    if data.header['OBSRVTRY'][i] == 'STEREO_B':
        data[..., i] /= calibration_ba[data.header['WAVELNTH'][i]]
# make sure it is 64 bits data
data.header['BITPIX'][:] = -64
# cube
shape = 3 * (256,)
header = {'CRPIX1':128.,
          'CRPIX2':128.,
          'CRPIX3':128.,
          'CDELT1':0.0234375/2.,
          'CDELT2':0.0234375/2.,
          'CDELT3':0.0234375/2.,
          'CRVAL1':0.,
          'CRVAL2':0.,
          'CRVAL3':0.,}
cube = siddon.fa.zeros(shape, header=header)
# model
P, D, obj_mask, data_mask = siddon.models.srt(data, cube,
                                              obj_rmin=1.,
                                              obj_rmax=1.7,
                                              data_rmax=1.3,
                                              mask_negative=True
                                              )
# hyperparameters
hypers = cube.ndim * (1e-1, )
# inversion
t = time.time()
#b = data[data_mask == 0]
b = data.flatten()
#sol = lo.quadratic_optimization(P, b, D, hypers, maxiter=100)
sol = lo.acg(P, b, D, hypers, maxiter=100)
print(time.time() - t)
# reshape result
fsol = siddon.fa.asfitsarray(sol.reshape(cube.shape), header=header)
fsol.tofits(os.path.join(siddon_path, "output", "test_siddon_secchi_mask.fits"))
