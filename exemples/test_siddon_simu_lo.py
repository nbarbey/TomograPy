#!/usr/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
import lo
# object
obj = siddon.siddon.centered_cubic_map(3, 32)
obj[:] = siddon.phantom.shepp_logan(obj.shape)
# data
radius = 200.
a = siddon.siddon.fov(obj.header, radius)
data = siddon.siddon.centered_stack(a, 128, n_images=60, radius=radius,
                                    max_lon=np.pi)
# projector
P = siddon.siddon_lo(data.header, obj.header)
# projection
t = time.time()
data = siddon.projector(data, obj)
print("projection time : " + str(time.time() - t))
# data
y = data.flatten()
# backprojection
t = time.time()
x0 = P.T * y
bpj = x0.reshape(obj.shape)
print("projection time : " + str(time.time() - t))
# coverage map
weights = (P.T * np.ones(y.size)).reshape(obj.shape)
# priors
Ds = [lo.diff(obj.shape, axis=i) for i in xrange(3)]
hypers = 1e-2 * np.ones(3)
#Ds, hypers = [], []
# inversion using scipy.sparse.linalg
t = time.time()
tol = 1e-6
sol = lo.acg(P, y, Ds, hypers,  maxiter=100, tol=tol)
sol = sol.reshape(bpj.shape)
print("inversion time : " + str(time.time() - t))
