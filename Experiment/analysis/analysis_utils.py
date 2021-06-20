import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# os.chdir('../')

Z = norm.ppf

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
    elif fa_rate == 0:
        fa_rate = half_fa

    d_prime = Z(hit_rate) - Z(fa_rate)
    c = -(Z(hit_rate) + Z(fa_rate)) / 2
    #     print(f'Hit rate: \t {hit_rate} \nFalse Alarm rate: {fa_rate}')

    return d_prime, c

def d_prime_rates(hit_rate, fa_rate):
    d_prime = Z(hit_rate) - Z(fa_rate)
    c = -(Z(hit_rate) + Z(fa_rate)) / 2
    
    return d_prime, c

def load_data(f_names, psychophys=False, drop_dupes=False):
    """
    Loads data in logs folder unless run is passed as an excluded run.
    Automatically excludes 0 runs.

    exclude_run (int or array, optional): runs to exclude

    returns: dataframe containing runs

    """

    all_logs = pd.DataFrame([])

    for f in f_names:
        # skip 0 and exclude trials
        if f.split('/')[-2].split('_')[3] == '0':
            continue

        df = pd.read_table(f, keep_default_na=True)

        # drop ISI rows
        df = df.drop(df[(df.phase % 2 == 1) & (df.event_type == 'stim')].index)

        # change NaNs to 0
        df['duration'] = df['duration'].fillna(0)
        df['nr_frames'] = df['nr_frames'].fillna(0)

        # add columns for end times
        df['end'] = df.onset + df.duration
        df['end_abs'] = df.onset_abs + df.duration

        # add column for task and color
        df['task'] = f.split('/')[-2].split('_')[-2]
#         df['color'] = f.split('/')[-2].split('_')[-2]
        df['run'] = f.split('/')[-2].split('_')[3]

        all_logs = all_logs.append(df, ignore_index=True)
        
#     if drop_dupes:
#         all_logs.drop_duplicates(subset='trial_nr', inplace=True)

    return all_logs


def analyse_logs(all_logs, duration=1):
    """
    Analyses dataframe containing logs
    """

    runs = sorted([int(i) for i in all_logs.run.unique()])
    psychophysics = pd.DataFrame(columns=['run', 'task', 'hits', 'misses', 'fas', 'correjs', 'dprime', 'criterion'])
    d_per_diff = pd.DataFrame(columns=['run', 'task','difficulty','hit_rate','fa_rate','d_prime','criterion'])
    large_AF_corr = []
    small_AF_corr = []

    for this_run in runs:
        this_run = str(this_run)
        df = all_logs[all_logs['run'] == this_run]

        if df.task.iloc[0] == 'l':
            task = 'color_balance'
            baseline = 0.5
        elif df.task.iloc[0] == 's':
            task = 'fix_intensity'
            baseline = 0.5

        stim_df = df[df.event_type == 'stim']
        switch_loc = np.diff(stim_df[task], prepend=baseline) != 0
        switch_loc = stim_df[(switch_loc) & (stim_df[task] != baseline)].index  # drop values where color_balance is 0.5
        responses = df.loc[df.response == 'space']

        tp = sum([(abs(i - responses.onset) < duration).any() \
                  for i in stim_df.loc[switch_loc].end])  # true positives
        fn = len(switch_loc) - tp  # false negatives (missed switches)
        fp = len(responses) - tp  # false positives (responded with no switch)
        tn = len(stim_df) - len(switch_loc) - fn  # true negative

        # accuracy = (tp + tn) / (tp + fn + fp + tn)
        d, c = d_prime(tp, fn, fp, tn)

        difficulty = sorted(stim_df[task].unique())

        # add column for correctness for each stim
        stim_df['correct'] = np.nan
        hits = [(abs(i - responses.onset) < duration).any() \
                for i in stim_df.loc[switch_loc].end]
        misses = switch_loc[np.invert(hits)]
        hits = switch_loc[hits]

        # label hits and misses in stim_df
        stim_df.loc[hits, 'correct'] = stim_df.loc[hits, 'correct'].replace(np.nan, 1)
        stim_df.loc[misses, 'correct'] = stim_df.loc[misses, 'correct'].replace(np.nan, 0)

        # get closest stimulus corresponding to a response
        response_ids = [(abs(i - stim_df.end)).idxmin() for i in responses.onset]

        # find false alarm values
        fas = []
        for idx, j in enumerate(stim_df.loc[response_ids, 'correct'].isna().index):
            if (abs(j - stim_df[stim_df.correct == 1].index) <= 2).any() == False:
                fas.append(idx)
                stim_df.loc[response_ids[idx], 'correct'] = 0

        # [stim_df.loc[response_ids[i]] for i in fas]

        # set remaining nans as correct rejections
        cor_rej = stim_df[stim_df.loc[:, 'correct'].isna()].index
        stim_df.loc[cor_rej, 'correct'] = 1

        psychophysics = psychophysics.append({'run': this_run,
                                              'task': task,
                                              'hits': tp,
                                              'misses': len(switch_loc) - tp,
                                              'fas': fp,
                                              'correjs': tn,
                                              'dprime': d,
                                              'criterion': c, 
                                              'difficulties': difficulty}, ignore_index=True)

        # get proportion correct
        for i in difficulty:
            hit_rate = (sum(stim_df[stim_df[task] == i].correct) / len(stim_df[stim_df[task] == i]))
            if task == 'color_balance':
                if i != 0.5:
                    d_per_diff = d_per_diff.append({'run': this_run,
                                                    'task':task,
                                                    'difficulty':i,
                                                    'hit_rate': hit_rate,
                                                    'fa_rate': fp/(fp + tn)}, ignore_index=True)
                    large_AF_corr.append([i, hit_rate])
            elif task == 'fix_intensity':
                if i != 0.5:
                    d_per_diff = d_per_diff.append({'run': this_run,
                                                    'task': task,
                                                    'difficulty': i,
                                                    'hit_rate': hit_rate,
                                                    'fa_rate': fp / (fp + tn)}, ignore_index=True)
                    small_AF_corr.append([i, hit_rate])

    return psychophysics, d_per_diff, large_AF_corr, small_AF_corr