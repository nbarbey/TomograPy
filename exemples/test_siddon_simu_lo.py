#!/usr/bin/env python
import time
import numpy as np
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
# priors
Ds = [lo.diff(obj.shape, axis=i) for i in xrange(3)]
# inversion using scipy.sparse.linalg
t = time.time()
sol = lo.acg(P, y, Ds, 1e-2 * np.ones(3),  maxiter=100, tol=1e-20)
sol = sol.reshape(bpj.shape)
print("inversion time : " + str(time.time() - t))
