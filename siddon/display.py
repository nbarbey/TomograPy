import numpy as np
from matplotlib import pyplot as plt

def display_object(obj):
    plt.figure()
    n = obj.shape[-1]
    div = _max_divider(n)
    shape = (obj.shape[0] * div, obj.shape[1] * n / div)
    tmp = np.zeros(shape)
    for k in xrange(n):
        i = k % div
        j = k / div
        i *= obj.shape[0]
        j *= obj.shape[1]
        tmp[i:i + obj.shape[0], j:j + obj.shape[1]] = obj[..., k]
    plt.imshow(tmp)
    plt.show()

def _max_divider(n):
    divmax = np.ceil(np.sqrt(n))
    divs = []
    for i in xrange(divmax):
        if np.remainder(n, i) == 0:
            divs.append(i)
    return np.max(divs)
