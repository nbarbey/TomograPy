======================================
TomograPy : A python tomography package
======================================

❗❗❗ A new version of this repo is being developed as `solartom <https://github.com/jmbhughes/solartom>`_ ❗❗❗


What is TomograPy ?
===================

This is a fast parallelized tomography projector / backprojector.  It
originates from solar tomography application but could be use for
other applications. It uses the fast Siddon algorithm as its core.
The parallelization is done with OpenMP on the C part of the code.

Application in solar tomography
===============================

For solar astrophysics, TomograPy allows to perform solar tomographic
reconstructions of the corona. It can take SOHO or STEREO data and
output a 3-dimensional map of the corona. For now only
Extreme-Ultraviolet data is handled but handling of coronographic data
is planed along with handling of data from other spacecrafts.

Requirements
============

You need numpy, pyfits, fitsarray for siddon to run.

Documentation
=============

The documentation for TomograPy is embedded into the code as
docstrings.  To get started, take a look at tomograpy/__init__.py or
in IPython, just do :

>>> import tomograpy
>>> tomograpy?

Exemple
=======

For an exemple on how to use TomograPy, see
exemple/test_siddon_simu.py

For solar tomography, you need first to process data using
http://www.lmsal.com/solarsoft/. Then you can take a look at
exemple/siddon_secchi_mask.py or use directly the siddon/srt script.

Home page
=========

https://github.com/nbarbey/siddon
