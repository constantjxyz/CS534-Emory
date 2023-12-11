'''idea:
        get the setting csv, return a dictionary to the main function
        the dictionary can be delivered to other python files(like engine.py), and can be altered
'''

# import dependent module
import numpy as np
import pandas as pd
import os

# define the function 'get_param_dict'
'''input: the param directory(csv file)
   output: a dictionary, dict.keys() for parameter name, dict.values() for parameter specific settings
'''
def get_param_dict(param_dir):
    assert param_dir.split('.')[-1] == 'csv'
    file = pd.read_csv(param_dir)
    d = dict()
    for i in range(len(file)):
        name = file.loc[i, 'name']
        setting = file.loc[i, 'setting']
        d[name] = setting
    return d
