#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from numpy import get_include
from os.path import join

# generate sources from template
import os
pth = os.getcwd()
execfile(pth + os.sep + "siddon" + os.sep + "parse_templates.py")

names = ('C_siddon', 'C_siddon4d')

ext_modules = [Extension(join('siddon', '_' + name + suffix_str % siddon_dict), 
                         [join('siddon', name + suffix_str + '.c') % siddon_dict,],
                         include_dirs=[join(get_include(), 'numpy', )],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         ) for siddon_dict in siddon_dict_list for name in names]

setup(name='Siddon',
      version='0.2.2',
      description='Solar tomography and Siddon algorithm',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      url="http://nbarbey.dyndns.org/software/siddon.html",
      requires=['numpy', 'scipy', 'fitsarray', 'pyfits'],
      packages=['siddon'],
      ext_modules=ext_modules
      )
