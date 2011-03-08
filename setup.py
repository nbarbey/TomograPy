#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from numpy import get_include
from os.path import join

# generate sources from template
import os
import imp
pth = os.getcwd()
#execfile(pth + os.sep + "siddon" + os.sep + "parse_templates.py")
pt_filename = pth + os.sep + "tomograpy" + os.sep + "parse_templates.py"
pt = imp.load_source("pt", pt_filename)
pt.generate_sources()

ext_modules = [Extension(join('tomograpy', '_' + name + pt.suffix_str % siddon_dict),
                         [join('tomograpy', name + pt.suffix_str + '.c') % siddon_dict,],
                         include_dirs=[join(get_include(), 'numpy', )],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],
                         ) for siddon_dict in pt.siddon_dict_list for name in pt.names]

setup(name='TomograPy',
      version='0.3.1',
      description='Solar tomography and Siddon algorithm',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      url="http://nbarbey.dyndns.org/software/siddon.html",
      requires=['numpy', 'scipy', 'pyfits', 'fitsarray', 'lo'],
      packages=['tomograpy'],
      ext_modules=ext_modules
      )
