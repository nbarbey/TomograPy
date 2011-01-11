#!/usr/bin/env python
import time
import numpy as np
import siddon
import lo
# object
obj = siddon.siddon.centered_cubic_map(3, 64, fill=1.)
# number of images
n = 64
# reshape object for 4d model
obj4 = obj[..., np.newaxis].repeat(n, axis=-1)
obj4.header['NAXIS'] = 4
obj4.header['NAXIS4'] = obj4.shape[3]
obj4.header['CRVAL4'] = 0.

# data 
radius = 200
a = siddon.siddon.fov(obj.header, radius)
data1 = siddon.siddon.centered_stack(a, 64, n_images=n/2, radius=radius,
                                    max_lon=np.pi, fill=0.)
data2 = siddon.siddon.centered_stack(a, 64, n_images=n/2, radius=radius,
                                    min_lon=np.pi / 2., max_lon=1.5 * np.pi, fill=0.)
data = siddon.solar.concatenate((data1, data2))

# times
DT = 1000.
dt_min = 100.
dates = np.arange(n / 2) * DT / 2.
dates = np.concatenate(2 * (dates, ))
dates = [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime((t))) for t in dates]
for i in xrange(len(data.header)):
    data.header[i]['DATE_OBS'] = dates[i]

data = siddon.solar.sort_data_array(data)

# projection
t = time.time()
data = siddon.projector4d(data, obj4, obstacle="sun")
print("projection time : " + str(time.time() - t))
# backprojection
x0 = obj4.copy()
x0[:] = 0.
t = time.time()
x0 = siddon.backprojector4d(data, x0, obstacle="sun")
print("backprojection time : " + str(time.time() - t))

# model
kwargs = {'obj_rmin':1., #'obj_rmax':1.3,
          'mask_negative':False, 'dt_min':100}
P, D, obj_mask, data_mask = siddon.models.stsrt(data, obj, **kwargs)
# hyperparameters
hypers = (1e-1, 1e-1, 1e-1, 1e5)
# test time for one projection
b = data.ravel()
t = time.time()
u = P.T * b
print("time with index grouping : %f" % ((time.time() - t)))
# inversion
t = time.time()
sol = lo.acg(P, b, D, hypers, tol=1e-10, maxiter=100)
# reshape result
fsol = siddon.fa.asfitsarray(sol.reshape(obj_mask.shape), header=obj4.header)
print(time.time() - t)
#fsol.tofits('stsrt_test.fits')
