#!/usr/bin/env python
import time
import numpy as np
import siddon
# object
n_repeats = 2
n = 512
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':n, 'NAXIS2':n, 'NAXIS3':n,
          'CRPIX1':n / 2., 'CRPIX2':n / 2., 'CRPIX3':n / 2.,
          'CDELT1':1.5 / n, 'CDELT2':1.5 / n, 'CDELT3':1.5 / n,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
obj = siddon.simu.object_from_header(header)
#obj[:] = siddon.phantom.shepp_logan(obj.shape)
obj[:] = 1.
# data
image_header = {'n_images':64,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':128, 'NAXIS2':128,
                'CRPIX1':64, 'CRPIX2':64,
                'CDELT1':6e-5, 'CDELT2':6e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
# loop on image shape and cube shape
pj_times = list()
bpj_times = list()
cube_naxes = 256,
#cube_naxes = [128, 256, 512, 1024]
#images_naxes = 128,
images_naxes = [128, 256, 512, 1024]
for cube_naxis in cube_naxes:
    print('cube naxis ' + str(cube_naxis))
    for i in xrange(3):
        header['NAXIS' + str(i + 1)] = cube_naxis
        header['CRPIX' + str(i + 1)] = cube_naxis / 2.
        header['CDELT' + str(i + 1)] = 1.5 / cube_naxis
    obj = siddon.simu.object_from_header(header)
    for images_naxis in images_naxes:
        print('image naxis ' + str(images_naxis))
        for i in xrange(2):
            image_header['NAXIS' + str(i + 1)] = images_naxis
            image_header['CRPIX' + str(i + 1)] = images_naxis / 2.
            image_header['CDELT' + str(i + 1)] = 0.00768 / images_naxis
        data = siddon.simu.circular_trajectory_data(**image_header)
        data[:] = np.zeros(data.shape)
        # projection
        t = time.time()
        for i in xrange(n_repeats):
            data = siddon.projector(data, obj)
        pj_times.append((time.time() - t)/ float(n_repeats))
        print("projection time : " + str(pj_times[-1]))
        # backprojection
        #x0 = siddon.fa.zeros(obj.shape, header=header)
        t = time.time()
        for i in xrange(n_repeats):
            obj = siddon.backprojector(data, obj)
        bpj_times.append((time.time() - t) / float(n_repeats))
        print("backprojection time : " + str(bpj_times[-1]))
