"""
Define test cases to use in other test modules.
"""
from copy import copy
import numpy as np
import siddon
# metadata
dtypes = [np.float32, np.float64]
pshapes = [3., 3.]
shapes = [1, 16]
radius = 200.
max_lon = 2 * np.pi
date_obs = "2010-01-01T00:00:00Z"
# objects lists
object_headers64 = [siddon.centered_cubic_map_header(ps, s, dtype=np.float64) for ps, s in zip(pshapes, shapes)]
object_headers32 = [siddon.centered_cubic_map_header(ps, s, dtype=np.float32) for ps, s in zip(pshapes, shapes)]
object_headers = object_headers64 + object_headers32

# image pshapes
im_pshapes = [siddon.siddon.fov(obj_h, radius) for obj_h in object_headers64]
shapes = [1, 32]

#image lists
image_headers64 = [siddon.centered_image_header(ps, s, dtype=np.float64) for ps, s in zip(im_pshapes, shapes)]
image_headers32 = [siddon.centered_image_header(ps, s, dtype=np.float32) for ps, s in zip(im_pshapes, shapes)]
image_headers = image_headers64 + image_headers32

# complement image headers with usefull keyword for simulations
for h in image_headers:
    h['radius'] = radius
    h['max_lon'] = max_lon
    h['DATE_OBS'] = date_obs
