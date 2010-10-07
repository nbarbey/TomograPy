import os
import numpy as np

c_siddon_template = "siddon" + os.sep + "C_siddon.c.template"
c_siddon4d_template = "siddon" + os.sep + "C_siddon4d.c.template"
templates = (c_siddon_template, c_siddon4d_template)


#ctypes = ["float", "double"]
ctypes = {"float":"float32", "double":"float64",
          "int":"int16", "long":"int32"}
obstacles = {"none":None, "sun":"sun"}

siddon_dict_list = []
for ctype in ctypes:
    for obstacle in obstacles:
        tmp_dict = {}
        tmp_dict['ctype'] = ctype
        tmp_dict['obstacle'] = obstacle
        siddon_dict_list.append(tmp_dict)

def generate_sources():
    for replace_dict in siddon_dict_list:
        for template in templates:
            parse_template(template, replace_dict)

def parse_template(filename, values=None):
    f = open(filename, "r")
    txt = f.read()
    f.close()
    if values is not None:
        txt = txt % values
    out_filename = set_filename(filename, values)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    f_out = open(out_filename, "w")
    f_out.write(txt)

def set_filename(filename, values):
    filename_list = filename.split(os.extsep)[:-1]
    filename_list[-2] += get_suffix(values)
    return '.'.join(filename_list)

def reverse_dict(in_dict):
    out_dict = dict()
    for k in in_dict: out_dict[in_dict[k]] = k
    return out_dict

ctypes_inv = reverse_dict(ctypes)
obstacles_inv = reverse_dict(obstacles)

def get_suffix(values):
    return '_' + '_'.join(values.values())

def get_suffix_str(values):
    return "_" + "_".join(['%(' + key + ')s' for key in values])

suffix_str = get_suffix_str(siddon_dict_list[0])

if __name__ == '__main__':
    generate_sources()
