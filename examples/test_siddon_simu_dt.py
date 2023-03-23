#!/usr/bin/env python
import time
import numpy as np
import tomograpy
# object
obj = tomograpy.centered_cubic_map(3, 128, fill=1.)
# number of images
n = 20
# reshape object for 4d model
obj4 = obj.reshape(obj.shape + (1,)).repeat(n, axis=-1)
obj4.header['NAXIS'] = 4
obj4.header['NAXIS4'] = obj4.shape[3]
obj4.header['CRVAL4'] = 0.

# data 
radius = 200
a = tomograpy.fov(obj.header, radius)
data = tomograpy.centered_stack(a, 128, n_images=n, radius=radius,
                                    max_lon=np.pi)
data[:] = np.zeros(data.shape)
# projection
t = time.time()
data = tomograpy.projector4d(data, obj4)
print("projection time : " + str(time.time() - t))
# backprojection
x0 = obj4.copy()
x0[:] = 0.
t = time.time()
x0 = tomograpy.backprojector4d(data, x0)
print("backprojection time : " + str(time.time() - t))
