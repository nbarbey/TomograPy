"""
This packages implements tomography algorithms and utilities in
python. It is made of several modules

- siddon: is the core of the package, with a fast C/OpenMP
  implementation of the Siddon algorithm :
  See : http://adsabs.harvard.edu/abs/1985MedPh..12..252S

- simu: implements some utilities to perform simulations

- secchi: a module to load STEREO/SECCHI data with appropriate
  metadata

- phantom: To generate phantoms (Shepp-Logan, Modified Shepp Logan,
  and Yu Ye Wang phantoms).


"""

from siddon import *
import simu
import secchi
import phantom
import models

try:
    import lo
except ImportError:
    pass

if 'lo' in locals():
    from lo_wrapper import *

