from sys import argv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import ipdb as pdb

_, fn = argv

L=15
weights = [[] for _ in range(L)]
grads = [[] for _ in range(L)]

with open(fn, 'r') as fin:
    for line in fin:
        s1_ret = re.search(r'weight_([0-9]+).*([0-9]+\.[0-9]*)', line) 
        if s1_ret is not None:
            w_idx = int(s1_ret.group(1))
            w_val = float(s1_ret.group(2))
            weights[w_idx].append(w_val)

        s2_ret = re.search(r'l2_grad_([0-9]+).*([0-9]+\.[0-9]*)', line)
        if s2_ret is not None:
            g_idx = int(s2_ret.group(1))
            g_val = float(s2_ret.group(2))
            grads[g_idx].append(g_val)

img_dir='/home/hanzhang/Dropbox/research/ann/img'

weights = np.asarray(weights)
grads = np.asarray(grads)
n_iter = weights.shape[1]
if n_iter > 0:
    x = np.arange(n_iter)
    plt.close('all')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for li in range(L):
        ax1.plot(x, weights[li, :], label='p{}'.format(li))

    plt.savefig(os.path.join(img_dir,'weights.png'), 
        bbox_inches='tight', dpi=fig1.dpi)


    if grads.shape[1] != n_iter:
        print "NO grad val found." 
        plt.show()
        sys.exit()
else:
    n_iter = grads.shape[1]
    x = np.arange(n_iter)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for li in range(L):
    ax2.plot(x, grads[li, :], label='p{}'.format(li)) 
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax2.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.savefig(os.path.join(img_dir, 'grads.png'), 
    bbox_inches='tight', dpi=fig2.dpi)

plt.show()

