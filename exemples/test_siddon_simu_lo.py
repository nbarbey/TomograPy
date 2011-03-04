#!/usr/bin/env python
import time
import numpy as np
import tomograpy
import lo
# object
obj = tomograpy.centered_cubic_map(3, 32)
obj[:] = tomograpy.phantom.shepp_logan(obj.shape)
# data
radius = 200.
a = tomograpy.fov(obj.header, radius)
data = tomograpy.centered_stack(a, 128, n_images=60, radius=radius,
                                    max_lon=np.pi)
# projector
P = tomograpy.lo(data.header, obj.header)
# projection
t = time.time()
data = tomograpy.projector(data, obj)
print("projection time : " + str(time.time() - t))
# data
y = data.flatten()
# backprojection
t = time.time()
x0 = P.T * y
bpj = x0.reshape(obj.shape)
print("projection time : " + str(time.time() - t))
# priors
Ds = [lo.diff(obj.shape, axis=i) for i in xrange(3)]
# inversion using scipy.sparse.linalg
t = time.time()
sol = lo.acg(P, y, Ds, 1e-2 * np.ones(3),  maxiter=100, tol=1e-20)
sol = sol.reshape(bpj.shape)
print("inversion time : " + str(time.time() - t))
