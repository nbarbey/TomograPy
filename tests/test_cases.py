"""
Define test cases to use in other test modules.
"""
import numpy as np

# metadata
minimal_object_header =  {'SIMPLE':True,'BITPIX':-64,
                          'NAXIS1':1, 'NAXIS2':1, 'NAXIS3':1,
                          'CRPIX1':.5, 'CRPIX2':.5, 'CRPIX3':.5,
                          'CDELT1':1., 'CDELT2':1., 'CDELT3':1.,
                          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}

small_object_header =  {'SIMPLE':True,'BITPIX':-64,
                        'NAXIS1':16, 'NAXIS2':16, 'NAXIS3':16,
                        'CRPIX1':8., 'CRPIX2':8., 'CRPIX3':8.,
                        'CDELT1':1., 'CDELT2':1., 'CDELT3':1.,
                        'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}

small_object_header32 =  {'SIMPLE':True,'BITPIX':-32,
                        'NAXIS1':16, 'NAXIS2':16, 'NAXIS3':16,
                        'CRPIX1':8., 'CRPIX2':8., 'CRPIX3':8.,
                        'CDELT1':1., 'CDELT2':1., 'CDELT3':1.,
                        'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}

minimal_image_header = {'n_images':1,
                        'SIMPLE':True, 'BITPIX':-64,
                        'NAXIS1':1, 'NAXIS2':1,
                        'CRPIX1':0., 'CRPIX2':0.,
                        'CDELT1':1., 'CDELT2':1.,
                        'CRVAL1':0., 'CRVAL2':0.,
                        }

small_image_header = {'n_images':1,
                      'SIMPLE':True, 'BITPIX':-64,
                      'NAXIS1':16, 'NAXIS2':16,
                      'CRPIX1':8., 'CRPIX2':8.,
                      'CDELT1':1., 'CDELT2':1.,
                      'CRVAL1':0., 'CRVAL2':0.,
                      }

small_image_header32 = {'n_images':1,
                      'SIMPLE':True, 'BITPIX':-32,
                      'NAXIS1':16, 'NAXIS2':16,
                      'CRPIX1':8., 'CRPIX2':8.,
                      'CDELT1':1., 'CDELT2':1.,
                      'CRVAL1':0., 'CRVAL2':0.,
                      }

object_headers = [minimal_object_header, small_object_header, small_object_header32]
image_headers = [minimal_image_header, small_image_header, small_image_header32]

# complement image headers with usefull keyword for simulations
for h in image_headers:
    h['radius'] = 1e6
    h['max_lon'] = np.pi
