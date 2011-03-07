#!/usr/bin/env python

"""
Performs Solar Rotational Tomography from the command line.
"""
from models import srt, stsrt, thomson

def usage():
    print(__usage__)

__usage__ = """Usage: srt [options] path [output]

Options:

  -h --help          Show this help message and exit.
  --config           Config file name (default: srt_default.cfg).
                     Command line options overloads config file options.

  Data parameters:

  -b --bin           Bin factor of images.
  -s --time_step     Time step between two images of the same kind.
  -m --tmin          Temporal beginning of data set
  -x --tmax          Temporal end of data set
  -i --instrument    Instrument name(s).
  -t --telescop      Telescops name(s).

  Object parameters:

  --naxis            Object shape in pixels.
  --crpix            Position of the reference pixel in fraction of pixels.
  --cdelt            Size of a pixel in physical coordinates.
  --crval            Position of the reference pixel in physical coordinates.

  Masking parameters:

  --obj_rmin
  --obj_rmax
  --data_rmin
  --data_rmax
  -n --negative      Mask negative data values.

  Optimization options:

  --model            Linear model to use for the inversion.
  --optimizer        Name of optimization routine (from lo).
  --hyperparameters  Hyperparameters of the smoothness prior.
  --maxiter          Maximum iteration number.
  --tol              Tolerance.
  --dt_min           Minimal time under which projection are considered 
                     simultaneous.

  Other options

  --input            Optional input filename for starting point.
  --output           Output filename (default: srt.fts).

"""

options = "hb:s:m:x:i:o:d:n"

long_options = ["help", "config=", "bin=", "time_step=", "tmin=", "tmax=",
                "instrument=", "telescop=",
                "naxis=", "crpix=", "cdelt=", "crval=",
                "obj_rmin=", "obj_rmax=", "data_rmin=", "data_rmax=",
                "negative",
                "model=", "optimizer=", "hyperparameters=", "maxiter=", "tol=",
                "dt_min=",
                "input=", "output="]

model_dict = {"srt":srt, "stsrt":stsrt, "thomson":thomson}

def main():
    """Handle config file, options and perform computations accordingly."""
    import os, getopt, sys, ConfigParser

    # parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    # defaults
    script_path = sys.path[0]
    config_file = os.path.join(script_path, "srt_default.cfg")
    obj_params = dict()
    data_params = dict()
    mask_params = dict()
    opt_params = dict()
    mask_negative = False
    output = "srt.fts"
    # parse config file
    for o, a in opts:
        if o == "--config":
            config_file = a
    config = ConfigParser.RawConfigParser()
    config.read(config_file)
    data_params["instrume"] = parse_tuple(config.get("data", "instrument"))
    data_params["telescop"] = parse_tuple(config.get("data", "telescop"))
    data_params["bin_factor"] = config.getint("data", "bin")
    data_params["time_step"] = config.getfloat("data", "time_step")
    data_params["tmin"] = config.get("data", "tmin")
    data_params["tmax"] = config.get("data", "tmax")
    obj_params["naxis"] = parse_tuple_int(config.get("object", "naxis"))
    obj_params["crpix"] = parse_tuple_float(config.get("object", "crpix"))
    obj_params["cdelt"] = parse_tuple_float(config.get("object", "cdelt"))
    obj_params["crval"] = parse_tuple_float(config.get("object", "crval"))
    mask_params["obj_rmin"] = config.getfloat("masking", "obj_rmin")
    mask_params["obj_rmax"] = config.getfloat("masking", "obj_rmax")
    mask_params["data_rmin"] = config.getfloat("masking", "data_rmin")
    mask_params["data_rmax"] = config.getfloat("masking", "data_rmax")
    mask_params["mask_negative"] = config.getboolean("masking", "negative")
    opt_params["model"] = model_dict[config.get("optimization", "model")]
    opt_params["optimizer"] = config.get("optimization", "optimizer")
    opt_params["hypers"] = parse_tuple_float(config.get("optimization", "hyperparameters"))
    opt_params["maxiter"] = config.getint("optimization", "maxiter")
    opt_params["tol"] = config.getfloat("optimization", "tol")
    try:
        opt_params["dt_min"] = config.getfloat("optimization", "dt_min")
    except(ConfigParser.NoOptionError):
        opt_params["dt_min"] = data_params["time_step"] / 2.
    # parse arguments
    if len(args) == 0:
        usage()
        sys.exit()
    path = args[0]
    if len(args) > 1:
        output = args[1]
    # parse options
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        # data parameters
        elif o in ("-b", "--bin"):
            data_params["bin_factor"] = int(a)
        elif o in ("-s", "--time_step"):
            data_params["time_step"] = float(a)
        elif o in ("-m", "--tmin"):
            data_params["tmin"] = a
        elif o in ("-x", "--tmax"):
            data_params["tmax"] = a
        elif o in ("-i", "--instrument"):
            data_params["instrume"] = parse_tuple(a)
        elif o in ("-t", "--telescop"):
            data_params["telescop"] = parse_tuple(a)
        # object parameters
        elif o == "--naxis":
            obj_params["naxis"] = parse_tuple_int(a)
        elif o == "--crpix":
            obj_params["crpix"] = parse_tuple_float(a)
        elif o == "--cdelt":
            obj_params["cdelt"] = parse_tuple_float(a)
        elif o == "--crval":
            obj_params["crval"] = parse_tuple_float(a)
        # masking parameters
        elif o == "--obj_rmin":
            mask_params["obj_rmin"] = float(a)
        elif o == "--obj_rmax":
            mask_params["obj_rmax"] = float(a)
        elif o == "--data_rmin":
            mask_params["data_rmin"] = float(a)
        elif o == "--data_rmax":
            mask_params["data_rmax"] = float(a)
        elif o in ("-n", "--negtative"):
            mask_params["mask_negative"] = True
        # optimization parameters
        elif o == "--model":
            opt_params["model"] = model_dict[a]
        elif o == "--optimizer":
            opt_params["optimizer"] = optimize_dict[a]
        elif o == "--hyperparameters":
            opt_params["hypers"] = parse_tuple_float(a)
        elif o == "--maxiter":
            opt_params["maxiter"] = int(a)
        elif o == "--tol":
            opt_params["tol"] = float(a)
        elif o in ("--input"):
            opt_params["input"] = a
        elif o in ("--dt_min"):
            opt_params["dt_min"] = a
        # other parameters
        elif o in ("-o", "--output"):
            output = a
        elif o == "--config":
            pass # handled before
        else:
            assert False, "unhandled option"
    sol = inversion(path, obj_params, data_params, opt_params, mask_params)
    sol.tofits(output)

def inversion(path, obj_params, data_params, opt_params, mask_params):
    """
    Perform an inversion using given parameters.
    """
    import numpy as np
    import lo, siddon
    import fitsarray as fa
    import solar, models
    # data
    data = solar.read_data(path, **data_params)
    if data is None:
        return
    data = solar.sort_data_array(data)
    # create object
    obj = make_object(obj_params)
    # configuration persistency
    out_header = persistency_header(obj.header, data_params,
                                       mask_params, opt_params)
    # pop optimization parameters
    model = opt_params.pop("model")
    optimizer = opt_params.pop("optimizer")
    hypers = opt_params.pop("hypers")
    if opt_params.has_key("input"):
        opt_params["x0"] = fa.FitsArray(file=opt_params["input"])
    # model
    P, D, obj_mask, data_mask = model(data, obj, **mask_params)
    # apply masking to data
    data *= (1 - data_mask)
    data[np.isnan(data)] = 0.
    # inversion
    b = data.ravel()
    exec("sol = lo." + optimizer + "(P, b, D, hypers, **opt_params)")
    # reshape result
    sol = fa.asfitsarray(sol.reshape(obj_mask.shape), header=out_header)
    return sol

def persistency_header(object_header, data_params, mask_params, opt_params):
    """
    Store srt parameters into output header. This allows to know how a
    given map has been computed.
    """
    import fitsarray as fa
    # get full parameters dict for persistency
    full_params = dict(data_params, **mask_params)
    full_params.update(opt_params)
    # change model to its name
    full_params['model'] = full_params['model'].__name__
    # select 8 first character and convert to strings
    full_params_str = dict()
    for k in full_params:
        if hasattr(full_params[k], "__iter__"):
            if len(full_params[k]) == 1:
                full_params_str[k[:8]] = full_params[k][0].__str__()
            else:
                for i, p in enumerate(full_params[k]):
                    full_params_str[k[:7] + str(i + 1)] = p.__str__()
        else:
            full_params_str[k[:8]] = full_params[k].__str__()
    # save configuration to object header
    out_header = dict(object_header)
    out_header.update(full_params_str)
    out_header = fa.dict2header(out_header)
    return out_header

def params_from_header(h):
    """
    Generate a config dicts from header (with "persistency" keys)
    """
    from siddon import dict_to_array
    #
    obj_params = dict()
    for k in ("naxis", "crpix", "crval", "cdelt"):
        obj_params[k] = dict_to_array(h, k.upper())
    #
    data_params = dict()
    for k in ("obj_rmin", "obj_rmax", "data_rmin", "data_rmax", "negative"):
        try:
            data_params[k] = h[k.upper()[:8]]
        except KeyError:
            pass
    #
    opt_params = dict()
    for k in ("model", "optimizer", "maxiter", "tol"):
        opt_params[k] = h[k.upper()[:8]]
    opt_params["hyperparameters"] = dict_to_array(h, "HYPERS")
    #
    mask_params = dict()
    for k in ("tmin", "tmax", "time_step", "bin_factor"):
        try:
            mask_params[k] = h[k.upper()[:8]]
        except KeyError:
            pass
    for k in ("telescop", "instrument"):
        try:
            mask_params[k] = dict_to_array(h, k.upper()[:8])
        except ValueError:
            pass
    return obj_params, data_params, opt_params, mask_params

def file_to_config(filename, config_filename=None):
    """
    Convert info from header into configuration for srt inversion.

    Arguments
    ---------

    filename (str):
      The filename of the fits file (output of srt).

    config_filename (str, optional):
      If provided, the config is saved into this file.

    Returns
    -------

    config (ConfigParser.RawConfigParser instance): A configuration
      instance, only if config_filename is not provided.
    """
    import pyfits
    import ConfigParser
    # read data
    h = dict(pyfits.fitsopen(filename)[0].header)
    # convert to dictionaries
    obj_params, data_params, opt_params, mask_params = params_from_header(h)
    # generate config
    config = ConfigParser.RawConfigParser()
    # define sections
    config.add_section("object")
    config.add_section("data")
    config.add_section("masking")
    config.add_section("optimization")
    # fill in sections
    for k in obj_params:
        config.set("object", k, obj_params[k])
    for k in data_params:
        config.set("data", k, data_params[k])
    for k in opt_params:
        config.set("optimization", k, opt_params[k])
    for k in mask_params:
        config.set("masking", k, mask_params[k])
    if config_filename is not None:
        fp = file(config_filename, "w")
        config.write(fp)
        fp.close()
    return config

def parse_tuple(my_str):
    """
    Parse input parameters which can be tuples.
    """
    # remove any kind of parenthesis
    for c in (")", "]", "}"):
        my_str = my_str.rstrip(c)
    for c in ("(", "[", "{"):
        my_str = my_str.lstrip(c)
    # split tuple elements if any
    str_list = my_str.split(",")
    # remove trailing whitespaces
    str_list = [s.rstrip() for s in str_list]
    str_list = [s.lstrip() for s in str_list]
    return str_list

def parse_tuple_int(my_str):
    """
    Parse tuple and convert to int.
    """
    return [int(s) for s in parse_tuple(my_str)]

def parse_tuple_float(my_str):
    """
    Parse tuple and convert to float.
    """
    return [float(s) for s in parse_tuple(my_str)]

def make_object_header(obj_params):
    naxis = obj_params["naxis"]
    header = dict()
    header["NAXIS"] = len(naxis)
    header["BITPIX"] = -64
    for i in xrange(len(naxis)):
        header["NAXIS" + str(i + 1)] = obj_params["naxis"][i]
        header["CRPIX" + str(i + 1)] = obj_params["crpix"][i]
        header["CDELT" + str(i + 1)] = obj_params["cdelt"][i]
        header["CRVAL" + str(i + 1)] = obj_params["crval"][i]
    return header

def make_object(*kargs):
    import fitsarray as fa
    header = make_object_header(*kargs)
    return fa.fitsarray_from_header(header)

if __name__ == "__main__":
    main()
