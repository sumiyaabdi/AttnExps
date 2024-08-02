import numpy as np
import yaml
from scipy.stats import norm
Z = norm.ppf

def load_yaml(file_name):
    with open(file_name, 'r') as file:
        settings=yaml.safe_load(file)
    return settings

def psyc_stim_list(stim_range, n_stim, baseline):
    " Creates stim list for psychophysics task"
    stim_list = stim_range*int(n_stim/len(stim_range))
    [stim_list.append(i) for i in [baseline]*(n_stim-len(stim_list))] # add baseline trials to fill in the rest
    np.random.shuffle(stim_list)

    return stim_list

def get_stim_nr(trial,phase,stim_per_trial):
    phase = phase-1 if phase % 2 == 1 else phase
    return int(trial * stim_per_trial + (phase)/2)

# def create_dot_grid(xy,jitter, n_stim):

#     """

#     Creates an n-Darray of xy coordinates of dot grid for 
#     main attention (i.e. detection) task.

#     Parameters
#     ----------
#     xy (2D array)  : 
#     jitter (float) :
#     n_stim (int)    :


#     Returns
#     -------

#     xys (n,x,y):    array of length n-trials containing xy 
#                     coordinates for each dot
#     """

#     np.random.uniform(-jitter/2,jitter/2)


def create_stim_list(n, signal, values,midpoint):
    """
    Creates stim list for main attention (i.e. detection)
    task.

    Parameters
    ----------
    n (int) :   number of stimuli to be presented
    signal (int) :  number of target stimuli
    values (1D arr) :   array containing range of target values

    Returns
    -------
    stim_list (array):  array of length n, target stimuli spaced
                        between non-target trials.
    """

    targets = []
    baseline = np.ones(int(n - signal)) * midpoint

    for i in range(len(values)):
        targets = np.append(targets, (np.ones(int(signal / len(values))) * values[i]))
    np.random.shuffle(targets)

    ratio = int(len(baseline) / len(targets))
    locs = [i * ratio + np.random.choice(ratio) for i in range(len(targets))]
    stim_list = np.insert(baseline, locs, targets)

    while len(stim_list) != n:
        stim_list = np.insert(stim_list, 0, midpoint)

    return stim_list


def sigmoid(x,x0,k):
    y = np.array(1 / (1 + np.exp(-k*(x-x0))))
    return y

def weibull(x,x0,k,g,l):
    y = g +(1-g -l)*sigmoid(x,k)
    return y

def inv_sigmoid(y,x0,k):
    return x0 - (np.log((1/y)-1)/k)

def d_prime(hits, misses, fas, crs):
    """
    calculate d' from hits(tp), misses(fn), false
    alarms (fp), and correct rejections (tn)

    returns: d_prime
    """

    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    hit_rate = hits / (hits + misses)
    fa_rate = fas / (fas + crs)

    # avoid d' infinity
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    elif hit_rate == 0:
        hit_rate = half_hit

    if fa_rate == 1:
        fa_rate = 1 - half_fa
    elif fa_rate <= 0:
        fa_rate = half_fa

    d_prime = Z(hit_rate) - Z(fa_rate)
    c = -(Z(hit_rate) + Z(fa_rate)) / 2

    return d_prime, c
