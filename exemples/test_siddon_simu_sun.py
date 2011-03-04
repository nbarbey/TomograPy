#!/usr/bin/env python
import time
import numpy as np
import tomograpy
# object
obj = tomograpy.centered_cubic_map(3, 128, fill=1.)
# data
radius = 200.
a = tomograpy.fov(obj.header, radius)
data = tomograpy.centered_stack(a, 128, n_images=17, radius=200., max_lon=np.pi)
# projection
t = time.time()
data = tomograpy.projector(data, obj, obstacle="sun")
print("projection time : " + str(time.time() - t))
# backprojection
obj0 = tomograpy.centered_cubic_map(3, 128, fill=0.)
t = time.time()
obj0 = tomograpy.backprojector(data, obj0, obstacle="sun")
print("backprojection time : " + str(time.time() - t))
