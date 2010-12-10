#!/usr/bin/env python
import time
import numpy as np
import siddon
# object
object_header = siddon.siddon.centered_cubic_map_header(3, 128)
obj = siddon.simu.object_from_header(object_header, fill=1.)
# data
radius = 200.
a = siddon.siddon.fov(object_header, radius)
data = siddon.siddon.centered_stack(a, 128, n_images=17, radius=200., max_lon=np.pi)
# projection
t = time.time()
data = siddon.projector(data, obj, obstacle="sun")
print("projection time : " + str(time.time() - t))
# backprojection
t = time.time()
obj0 = siddon.simu.object_from_header(object_header, fill=0.)
obj0 = siddon.backprojector(data, obj0, obstacle="sun")
print("backprojection time : " + str(time.time() - t))
