#!/usr/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
import lo
# object
obj = siddon.siddon.centered_cubic_map(10, 64)
obj[:] = siddon.phantom.shepp_logan(obj.shape)
# data 
data = siddon.siddon.centered_stack(0.06144, 128, n_images=60, radius=200., max_lon=np.pi)
# model
kwargs = {"pb":"pb", "obj_rmin":1.5, "data_rmin":1.5}
P, D, obj_mask, data_mask = siddon.models.thomson(data, obj, u=.5, **kwargs)
# projection
t = time.time()
data[:] = (P * obj.ravel()).reshape(data.shape)
print("projection time : " + str(time.time() - t))
# data
# backprojection
t = time.time()
x0 = (P.T * data.ravel()).reshape(obj.shape)
bpj = x0.reshape(obj.shape)
print("backprojection time : " + str(time.time() - t))
# coverage map
weights = (P.T * np.ones(data.size)).reshape(obj.shape)
# hyperparameters
hypers = 1e-3 * np.ones(3)
# inversion using scipy.sparse.linalg
t = time.time()
tol = 1e-8
sol = lo.acg(P, data.ravel(), D, hypers,  maxiter=100, tol=tol)
sol = sol.reshape(obj.shape)
print("inversion time : " + str(time.time() - t))
