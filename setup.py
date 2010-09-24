#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from numpy import get_include
from os.path import join

setup(name='Siddon',
      version='0.2',
      description='Solar tomography and Siddon algorithm',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      requires=['numpy', 'scipy', 'fitsarray'],
      packages=['siddon'],
      ext_modules=[Extension(join('siddon', '_C_siddon'), 
                             [join('siddon','C_siddon.c')],
                             include_dirs=[join(get_include(), 'numpy', )],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             )],
      )
