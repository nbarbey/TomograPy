"""
This packages implements tomography algorithms and utilities in
python. It is made of several modules, as follows:

- siddon: The core of the package, with a fast C/OpenMP
  implementation of the Siddon algorithm.
  See: http://adsabs.harvard.edu/abs/1985MedPh..12..252S

- simu: Implements some utilities to perform simulations.

- solar: A module to load Solar physics data with appropriate
  metadata.

- models: Defines linear model linking the data with the parameters
  to estimate (emission maps, density maps).

    * srt: Solar rotational tomography

    * stsrt: Smooth temporal solar rotational tomography

    * thomson: srt with Thomson scattering model (e.g. for white light data)

- phantom: To generate phantoms (Shepp-Logan, Modified Shepp Logan,
  and Yu Ye Wang phantoms).

Along with these modules, a convenient way to call inversion routines
is provided in the command-line through the srt command. In order to
use srt, you need to have the srt file in your $PATH along with
siddon, lo and fitsarray in your $PYTHONPATH. For usage information,
type "srt -h".

"""

from siddon import *
import simu
import solar
import phantom
import models
import display
import srt as srt_cli

try:
    import lo
except ImportError:
    pass

if 'lo' in locals():
    from lo_wrapper import *


version = "0.3.0"
