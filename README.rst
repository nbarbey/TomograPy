====================================
Siddon : A python tomography package
====================================

What is Siddon ?
================

This is a fast parallelized tomography projector / backprojector.  It
originates from solar tomography application but could be use for
other applications. It uses the fast Siddon algorithm as its core.
The parallelization is done with OpenMP on the C part of the code.

Requirements
============

You need numpy, pyfits, fitsarray for siddon to run.

Exemple
=======

For an exemple on how to use siddon, see exemple/test_siddon_simu.py

Home page
=========

http://nbarbey.dyndns.org/software/siddon.html
