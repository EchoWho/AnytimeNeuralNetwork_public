from sys import argv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import ipdb as pdb

if len(argv) > 2:

    _, fn, dey_log_dir, dey_log_basename = argv
else:
    _, fn = argv

    dey_log_dir='/home/debadeepta/ann_models_logs'
    dey_log_basename='cust-p-run_exp_'

exceptions = [1, 36, 83] +[71,72,73, 75,76,77]

with open(fn, 'r') as fin:
    for li, line in enumerate(fin):
        if len(line) > 1:
            #print line
            dir_name = os.path.join(dey_log_dir, 
                                    '{}{}'.format(dey_log_basename, li+1))
            #print dir_name
            if not os.path.isdir(dir_name) and not li+1 in exceptions:
                print 'exp id: {} does not exist, but is expected. Exp description: {}'.format(li+1, line[:-1])
            
