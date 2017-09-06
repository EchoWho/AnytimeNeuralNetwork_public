import numpy as np

__all__ = ['sieve_loss_weights',  'eann_sieve', 
    'optimal_at',
    'exponential_weights', 'at_func', 
    'constant_weights', 'stack_loss_weights',
    'half_constant_half_optimal', 'linear',
    'recursive_heavy_end', 
    'quater_constant_half_optimal',
    'loss_weights']

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
    return np.ones(N,dtype=np.float32) #/ N * np.log2(N)

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
    """
    Not sure what this was doing... emphasize the end and the start
    """
    weights = np.ones(N, dtype=np.float32)
    if N <= 2: 
        return weights
    weights[-1] = 2*N-4
    weights[0] = N-2
    weights /= np.float(4 * N - 8)
    return weights

def recursive_heavy_end(N):
    """
    N, N/2, N/4,... are set to have 1/3 of the total weights up to
    its depth. 
    N, N/2, N/4,... have weights decay exponentially

    The other weights have constant weight
    """
    weights = np.ones(N, dtype=np.float32)
    i = N-1
    w = 1.0 * N
    while True:
        weights[i] += w
        if i == 0:
            break
        i = i // 2
        w = w / 2.0 
    weights[-1] += N # make sure last layer has 1/2
    weights /= np.sum(weights) / np.log2(N)
    return weights

def linear(N, a=0.25, b=1.0):
    delta = (b-a) / (N-1.0)
    weights = np.arange(N, dtype=np.float32) * delta + a
    #weights /= np.sum(weights) / np.log2(N)
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

def loss_weights(N, args):
    FUNC_TYPE = args.func_type
    if FUNC_TYPE == 0: # exponential spacing
        return at_func(N, func=lambda x:2**x)
    elif FUNC_TYPE == 1: # square spacing
        return at_func(N, func=lambda x:x**2)
    elif FUNC_TYPE == 2: #optimal at ?
        return optimal_at(N, args.opt_at)
    elif FUNC_TYPE == 3: #exponential weights
        return exponential_weights(N, base=args.exponential_base)
    elif FUNC_TYPE == 4: #constant weights
        return constant_weights(N) 
    elif FUNC_TYPE == 5: # sieve with stack
        return stack_loss_weights(N, args.stack, sieve_loss_weights)
    elif FUNC_TYPE == 6: # linear
        return linear(N, a=0.25, b=1.0)
    elif FUNC_TYPE == 7: # half constant, half optimal at -1
        return half_constant_half_optimal(N, -1)
    elif FUNC_TYPE == 8: # quater constant, half optimal
        return quater_constant_half_optimal(N)
    elif FUNC_TYPE == 9: # recursive heavy end
        return stack_loss_weights(N, args.stack, recursive_heavy_end) 
    else:
        raise NameError('func type must be either 0: exponential or 1: square' \
            + ' or 2: optimal at --opt_at, or 3: exponential weight with base --base')


