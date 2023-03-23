#!/usr/bin/env python
import time
import numpy as np
import tomograpy
# object
object_header = tomograpy.centered_cubic_map_header(3, 128)
obj = tomograpy.simu.object_from_header(object_header, fill=1.)
# data
radius = 200.
a = tomograpy.fov(object_header, radius)
data = tomograpy.centered_stack(a, 128, n_images=60, radius=200., max_lon=np.pi)
# projection
t = time.time()
data = tomograpy.projector(data, obj)
print("projection time : " + str(time.time() - t))
# backprojection
t = time.time()
data[:] = 1.
obj0 = tomograpy.simu.object_from_header(object_header, fill=0.)
obj0 = tomograpy.backprojector(data, obj0)
print("backprojection time : " + str(time.time() - t))

obj1 = tomograpy.simu.object_from_header(object_header, fill=0.)
obj1 = tomograpy.backprojector(data, obj1)
