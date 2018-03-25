#precedes globals.py
import numpy as np
#import tensorflow as tf
import pandas as pd
from collections import Counter
from pytil.utility import *
from pytil.object import Namespace as O


def clamp(value, min, max):
    '''
    Clamp value between min and max. Asserts min < max.
    '''
    assert min <= max
    if value < min:
        return min
    if value > max:
        return max
    return value


class Incrementer:
    def __init__(self, start=0):
        self.start = start
        
    def __call__(self):
        temp = self.start
        self.start += 1
        return temp


def cdf(samples):
    '''    
    Paramaters
    ----------
    samples : iterable of sample values of random variable

    Return
    ------
    x, y : xy-coordinates of observed CDF plot
    '''
    x = np.sort(samples)
    y = np.array(range(len(samples))) / float(len(samples))
    return x, y


def merge_axes(x, at, to):
    '''
    Return reshaped x where the axes inclusively between indices at and to are merged into one axis.
    e.g. assert x.shape == [2, 3, 5] -> x = merge_axes(x, 1, 2) -> assert x.shape == [2, 15]
    '''
    t = x.shape
    return x.reshape(t[:at] + (prod(t[at:(to + 1)]),) + t[(to + 1):])


def rolling_window(time_series, window_size):
    '''    
    Paramaters
    ----------
    time_series : tf.Tensor or np.ndarray of shape (<number of observations>, [...])
    window_size : int

    Return
    ------
    tf.Tensor or np.ndarray of shape (<number of windows>, window_size)
    '''
    if isinstance(time_series, tf.Tensor):
        windows = [time_series[i : i + window_size] for i in range(time_series.shape[0] + 1 - window_size)]
        return tf.stack(windows)
    else:
        n_obs = len(time_series)
        return np.lib.stride_tricks.as_strided(time_series,
                                               shape = (n_obs + 1 - window_size, window_size) + time_series.shape[1:],
                                               strides = (time_series.strides[0],) + time_series.strides)


def confusion(targets, guesses):
    '''    
    Paramaters
    ----------
    targets : np.ndarray
    guesses : np.ndarray

    Return
    ------
    confusion table as a pandas.DataFrame
    '''
    y_true = targets.astype(int)
    y_pred = np.round(guesses).astype(int)
    return pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


def multi_indices(lengths):
    '''    
    Paramaters
    ----------
    lengths : iterable of integer sizes

    Return
    ------
    generator that yields the multi-indices for all entries of a lengths shaped tensor
    '''
    if prod(lengths) == 0:
        return
    n = len(lengths)
    ii = n * [0]
    c = n - 1
    k = 0
    while True:
        if ii[c] == lengths[c]:
            if c == 0:
                break
            ii[c] = 0
            c -= 1
            ii[c] += 1
            continue
        yield tuple(ii)
        c = n - 1
        ii[c] += 1


def listify(seq_of_values):
    out = []
    for x in seq_of_values:
        out.append([x])
    return out


def cross_combine_2_sequences(heads, tails):
    for head in heads:
        for tail in tails:
            out = list(head)
            out.extend(tail)
            yield out

def cross_combine_sequences(*parts):
    if len(parts) == 0:
        return []
    elif len(parts) == 1:
        return parts[0]
    else:
        head = parts[0]
        for tail in parts[1:]:
            head = cross_combine_2_sequences(head, tail)
        return head
        

def recall(confusion):
    con = confusion
    if isinstance(con, pd.DataFrame):
        con = con.values[:2, :2]
    return (con[0, 0] / (con[0, 0] + con[0, 1]), con[1, 1] / (con[1, 1] + con[1, 0]))

def precision(confusion):
    con = confusion
    if isinstance(con, pd.DataFrame):
        con = con.values[:2, :2]
    return (con[0, 0] / (con[0, 0] + con[1, 0]), con[1, 1] / (con[1, 1] + con[0, 1]))

def accuracy(confusion):
    con = confusion
    if isinstance(con, pd.DataFrame):
        con = con.values[:2, :2]
    return (con[0, 0] + con[1, 1]) / (con[0, 0] + con[0, 1] + con[1, 0] + con[1, 1])


def consec(a, e, v=None):
    if v is None:
        v = a
    agg = [0]
    aggv = [[]]
    for x, c in zip(a, v):
        if x == e:
            agg[-1] += 1
            aggv[-1].append(c)
        elif agg[-1] != 0:
            agg.append(0)
            aggv.append([])
    if agg[-1] == 0:
        agg.pop()
        aggv.pop()
    return agg, aggv

def hist_consec(a, e, v=None):
    agg, aggv = consec(a, e, v)
    if v is None:
        cnt = Counter(agg)
        return [list(_) for _ in zip(*sorted((n, c * n) for n, c in cnt.items()))]
    hist = Counter()
    for n, values in zip(agg, aggv):
        hist[n] += sum(values)
    ns = sorted(set(agg))
    h = [hist[n] for n in ns]
    return ns, h

def cumhist_consec(a, e, v=None):
    x, h = hist_consec(a, e, v)
    x_lookup = {n: i for i, n in enumerate(x)}
    p = []
    for n in range(1, max(x) + 1):
        prev = p[-1] if p else 0
        try:
            i = x_lookup[n]
            p.append(prev + h[i])
        except KeyError:
            p.append(prev)
    return list(range(1, max(x) + 1)), p


def max_drawdown(a):
    h = -float("inf")
    ans = -float("inf")
    for x in a:
        ans = max(ans, h - x)
        h = max(h, x)
    return ans


def hist(a, *args, weighted=False, **kw):
    if weighted:
        kw['weights'] = a.abs()
    return a.hist(*args, **kw)


'''
class hp_distributions(metaclass=O):
    # meta
    seed = lambda: hash(random.random())
    # booster params
    booster = lambda: random.choices(["gbtree", "dart"], [1, 3])[0]
    eta = lambda: clamp(random.expovariate(8), 0, 1)
    gamma = 0
    max_depth = lambda: random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [.3, 1, 1, 1, .3, .2, .2, .1, .1, .1])
    min_child_weight = lambda: random.expovariate(10 ** random.choice(1, 0, -1, -2))
    max_delta_step = 0
    subsample = 
    colsample_bytree = 1
    colsample_bylevel = 1
    reg_lambda = 1
    reg_alpha = 1
    # dart:
    sample_type = 'uniform'
    normalize_type = 'tree'
    rate_drop = 0
    one_drop = 0
    skip_drop = 0
    # learning function params
    num_boost_round = 10
    early_stopping_rounds = 12
'''

#base_results = [{hp_values: [None for _ in range(ml_config.n_test_years)] for hp_values in hp_values_search_list} for target in targets]
# base_results[i_target][hp_values][n_years_from_present] = result
#for i_target, y in enumerate(targets):
#    for i_hp_values, hp_values in enumerate(hp_values_search_list):
#        for n_years_from_present in range(ml_config.n_test_years):
#            pass        