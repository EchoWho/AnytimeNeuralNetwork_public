import numpy as np

__all__ = ['sieve_loss_weights', 'stack_loss_weights']


def sieve_loss_weights(N):
    log_n = int(np.log2(N))
    weights = np.zeros(N)
    for j in range(log_n + 1):
        t = int(2**j)
        wj = [ 1 if i%t==0 else 0 for i in range(N) ] 
        weights += wj
    weights[0] = np.sum(weights[1:])
    weights = weights / np.sum(weights) * log_n
    return weights

def stack_loss_weights(N, stack, method=sieve_loss_weights):
    weights = np.zeros(N)
    weights[0:N:stack] = method(N//stack)
    return weights
