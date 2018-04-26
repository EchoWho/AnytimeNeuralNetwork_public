
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import re


model_dir = '/home/hanzhang/models/ann_policy/cifar100/'

def model_name_to_cost(name):
    re_ret = re.search(r'n([0-9]*)-c([0-9]*)-.*', name)
    n = float(re_ret.group(1).strip())
    c = float(re_ret.group(2).strip())

    cost = (c / 16) **2  * n / 15
    return cost

l_models = sorted([
        'n17-c16-s6', 'n17-c32-s6', 
        'n33-c16-s12', 'n33-c32-s12',  
        'n5-c16-s2', 'n5-c32-s2',
        'n65-c16-s24', 'n65-c32-s24',
        'n9-c16-s3', 'n9-c32-s3'], key=model_name_to_cost)

l_xs = []
l_ys = []
l_costs = [model_name_to_cost(name) for name in l_models ]

for model_nm in l_models:
    name = os.path.join(model_dir, model_nm, '{}-cifar100-feats-preds.npz'.format(model_nm))
    d = np.load(name)

    if model_nm == 'n5-c32-s2':
        xy_name = os.path.join(model_dir, model_nm, '{}-cifar100-feats-preds_XY.npz'.format(model_nm))
        d_xy = np.load(xy_name)

        X = d_xy['l_images']
        Y = d_xy['l_labels']

    x_m = d['l_feats']
    y_m = d['l_preds']

    l_xs.append(x_m)
    l_ys.append(y_m)

    
#accuracies:
for mi, _y in enumerate(l_ys):
    accu = np.sum(Y == np.argmax(_y, axis=1)) / 5000.
    print 'model {}; cost {} ; accu {}'.format(mi, l_costs[mi], accu)

def compute_targets(lam):
    # per sample best:
    l_tars = []
    l_reg_tars = []
    for i, y in enumerate(Y):
        min_tar = np.inf
        min_tar_i = 0
        # log loss, oops this is wrong: the logits are not normalized.
        #tars = [-l_ys[mi][i][y] + lam * l_costs[mi] for mi in range(len(l_models))]

        # maximization objective
        tars = [ np.float32( np.argmax(Y_hat[i]) == y ) - lam * cost \
                    for Y_hat, cost in zip(l_ys, l_costs) ]
        tar_i = np.argmax(tars)
        l_tars.append(tar_i)
        l_reg_tars.append(tars)
    l_tars = np.asarray(l_tars, dtype=int)
    l_reg_tars = np.asarray(l_reg_tars, dtype=np.float32)
    return l_tars, l_reg_tars


full_X = np.concatenate(l_xs, axis=1)
#full_X = np.concatenate([full_X, flat_X], axis=1)


l_tars, l_reg_tars = compute_targets(lam=0.1)
print  np.histogram(np.argmax(l_reg_tars, axis=1), list(range(len(l_costs) + 1)))
class_counts = np.histogram(l_tars, list(range(len(l_costs) + 1)))
np.savez('/home/hanzhang/data/ann_policy/targets_cifar100.npz', targets=l_tars, counts=class_counts)


mean, std = np.mean(l_reg_tars, axis=0), np.std(l_reg_tars, axis=0)
np.savez('/home/hanzhang/data/ann_policy/reg_targets_cifar100.npz', reg_targets=l_reg_tars, mean=mean, std=std)



#rfc = RandomForestClassifier(max_depth=5, n_estimators=100)
#rfc.fit(full_X[:4500,:], l_tars[:4500])

#preds = rfc.predict(full_X[4500:,:]
