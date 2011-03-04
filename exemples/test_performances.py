#!/usr/bin/env python
import time
import numpy as np
import tomograpy
# object
n_repeats = 2
# loop on image shape and cube shape
pj_times = list()
bpj_times = list()
cube_naxes = 256,
#cube_naxes = [128, 256, 512, 1024]
#images_naxes = 128,
images_naxes = [128, 256, 512, 1024]
for cube_naxis in cube_naxes:
    print('cube naxis ' + str(cube_naxis))
    obj = tomograpy.centered_cubic_map(3, cube_naxis)
    for images_naxis in images_naxes:
        print('image naxis ' + str(images_naxis))
        for i in xrange(2):
            data = tomograpy.centered_stack(0.0075, images_naxis,
                                                n_images=64, radius=200.,
                                                max_lon=np.pi)
        # projection
        t = time.time()
        for i in xrange(n_repeats):
            data = tomograpy.projector(data, obj)
        pj_times.append((time.time() - t)/ float(n_repeats))
        print("projection time : " + str(pj_times[-1]))
        # backprojection
        #x0 = tomograpy.fa.zeros(obj.shape, header=header)
        t = time.time()
        for i in xrange(n_repeats):
            obj = tomograpy.backprojector(data, obj)
        bpj_times.append((time.time() - t) / float(n_repeats))
        print("backprojection time : " + str(bpj_times[-1]))
