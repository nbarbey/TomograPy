import numpy as np
from . import axpy_py

def print_hello(name):
    print(f"hello {name}")

def example():
    a = 3
    x = np.arange(10).astype(float)
    y = np.ones(10).astype(float)
    return axpy_py(a, x, y)
