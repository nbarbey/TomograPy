#!/usr/bin/env python

"""
Small projection test to compare with IDL siddon.
"""

import numpy as np
import siddon
im = siddon.siddon.centered_stack(0.0016, 32, n_images=1, radius=200., fill=0.)
cube = siddon.siddon.centered_cubic_map(3, 256, fill=1.)
P = siddon.siddon_lo(im.header, cube.header, obstacle="sun")
im[:] = (P * cube.ravel()).reshape(im.shape)
