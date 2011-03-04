#!/usr/bin/env python
import time
import numpy as np
import tomograpy
import lo
# object
obj = tomograpy.centered_cubic_map(10, 64)
obj[:] = tomograpy.phantom.shepp_logan(obj.shape)
# data 
radius = 200
a = tomograpy.fov(obj, radius)
data = tomograpy.centered_stack(a, 128, n_images=60, radius=radius, max_lon=np.pi)
# model
kwargs = {"pb":"pb", "obj_rmin":1.5, "data_rmin":1.5}
P, D, obj_mask, data_mask = tomograpy.models.thomson(data, obj, u=.5, **kwargs)
# projection
t = time.time()
data[:] = (P * obj.ravel()).reshape(data.shape)
print("projection time : " + str(time.time() - t))
# data
# backprojection
t = time.time()
x0 = P.T * data.ravel()
bpj = x0.reshape(obj.shape)
print("backprojection time : " + str(time.time() - t))
# inversion using scipy.sparse.linalg
t = time.time()
sol = lo.acg(P, data.ravel(), D, 1e-3 * np.ones(3),  maxiter=100, tol=1e-8)
sol = sol.reshape(obj.shape)
print("inversion time : " + str(time.time() - t))
