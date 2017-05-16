import numpy as np

__all__ = ['sieve_loss_weights',  'eann_sieve', 
    'optimal_at',
    'exponential_weights', 'at_func', 
    'constant_weights', 'stack_loss_weights',
    'half_constant_half_optimal', 'linear',
    'quater_constant_half_optimal']

def sieve_loss_weights(N):
    if N == 1:
        return np.ones(1)
    num_steps = 0
    step = 1
    delt_weight = 1.0
    weights = np.zeros(N)
    while step < N:
        weights[0:N:step] += delt_weight
        step *= 2
        num_steps += 1
    weights[0] = np.sum(weights[1:])
    weights /= (np.sum(weights) / num_steps)
    return np.flipud(weights)

def eann_sieve(N):
    weights = sieve_loss_weights(N)
    weights[:N//2] = 0.0
    return weights

def constant_weights(N):
    return np.ones(N,dtype=np.float32)

def optimal_at(N, optimal_l):
    """ Note that optimal_l is zero-based """
    weights = np.zeros(N)
    weights[optimal_l] = 1.0
    return weights

def half_constant_half_optimal(N, optimal_l=-1):
    weights = np.ones(N, dtype=np.float32)
    if N > 1:
        weights[optimal_l] = N-1
    weights /= np.float(N-1)
    return weights

def quater_constant_half_optimal(N):
    weights = np.ones(N, dtype=np.float32)
    if N <= 2: 
        return weights
    weights[-1] = 2*N-4
    weights[0] = N-2
    weights /= np.float(4 * N - 8)
    return weights

def linear(N, a=0.25, b=1.0):
    delta = (b-a) / (N-1.0)
    weights = np.arange(N, dtype=np.float32) * delta + a
    return weights

def exponential_weights(N, base=2.0):
    weights = np.zeros(N, dtype=np.float32)
    weights[0] = 1.0
    for i in range(1,N):
        weights[i] = weights[i-1] * base
    if base >= 1.0:
        max_val = weights[-1]
    else:
        max_val = weights[0]
    weights /= max_val / int(np.log2(N))
    return weights

def at_func(N, func=lambda x:x, method=sieve_loss_weights):
    pos = []
    i = 0
    do_append = True
    while do_append:
        fi = int(func(i))
        if fi >= N:
            do_append = False
            break
        pos.append(fi)
        i += 1
    if len(pos) == 0 or pos[-1] != N-1:
        pos.append(N-1)
    #elif pos[-1] != N-1:
    #    pos = (N-1-pos[-1]) + np.asarray(pos) 
    weights = np.zeros(N, dtype=np.float32)
    weights[pos] = method(len(pos))
    return weights

def stack_loss_weights(N, stack, method=sieve_loss_weights):
    weights = np.zeros(N, dtype=np.float32)
    weights[(N-1)%stack:N:stack] = method(1+(N-1)//stack)
    return weights
