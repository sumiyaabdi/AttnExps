import yaml
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
import numpy as np
import os
opj = os.path.join
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

Z = norm.ppf

##  UNDER CONSTRUCTION ##

class AnalyseRun():

    def __init__(self,folder,task,attn):
        self.folder=folder
        self.task=task
        self.attn=attn
        self.wd = os.getcwd()

        settings_f = opj(self.wd, f'expsettings/expsettings_{self.task}.yml')
        with open(settings_f) as file:
            self.settings = yaml.safe_load(file)

    def analyse2afc(self):

        f = glob.glob(f"{self.wd}/logs/{self.folder}_Logs/*.tsv")[0]

        sub,ses,task,run = [i.split('-')[-1] for i in self.folder.split('_')]
        sz = task[-1]
        task=task[:-1]
        resp = self.settings['attn_task']['resp_keys']
        resp_blue, resp_pink = [str(i).upper() for i in resp]

        df = pd.read_table(f,keep_default_na=True)
        df = df[df.event_type == 'response']
        df.drop(['nr_frames','duration'],axis=1,inplace=True)
        df['response'] = df.response.astype(str)
        df['attn_size']=sz
        df['corr_L']=np.nan
        df['corr_S']=np.nan

        cols = [['large_prop','small_prop'],['corr_L','corr_S']]
        self.summary = pd.DataFrame([])
        not_task = pd.DataFrame([])

        for prop,cor in zip(cols[0],cols[1]):
            true_blue = df[(df[prop] > 0.5) & (df.response == resp_blue)].index
            false_blue = df[(df[prop] < 0.5) & (df.response == resp_blue)].index
            true_pink = df[(df[prop] < 0.5) & (df.response == resp_pink)].index
            false_pink = df[(df[prop] > 0.5) & (df.response == resp_pink)].index

            df.loc[np.hstack((true_blue,true_pink)), cor] = 1
            df.loc[np.hstack((false_blue,false_pink)), cor] = 0
            df =  df.dropna(subset=[cor])

            diffs = sorted(df[prop].unique())
            task_summary = pd.DataFrame([])

            for idx, diff in enumerate(diffs):
                task_summary.loc[idx,'attn_size'] = prop.split('_')[0][0]
                task_summary.loc[idx,'diff'] = diff
                task_summary.loc[idx,'n_trials'] = len(df[df[prop] == diff])
                task_summary.loc[idx,'resp_blue'] = len(df[(df[prop] == diff ) & (df.response == resp_blue)])/len(df[df[prop] == diff])
                task_summary.loc[idx,'n_correct'] = len(df[(df[prop] == diff) & (df[cor] == 1)])
                task_summary.loc[idx,'percent_correct'] = 100*len(df[(df[prop] == diff) & (df[cor] == 1)])/len(df[df[prop] == diff])

            self.summary = self.summary.append(task_summary, ignore_index=True)

    def plot2afc(self):
        # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # fig.suptitle(f'ATTENTION SIZE: {self.attn.upper()}', fontsize=16)

        # axs[0].set_title('SmallAF Performance')
        # axs[0].set_ylim(0, 1)
        # axs[0].set_ylabel('Response Blue')
        # axs[0].set_xlabel('% Blue')
        # axs[0].scatter(self.summary[self.summary.attn_size == 's']['diff'],\
        #                self.summary[self.summary.attn_size == 's']['resp_blue'])

        # axs[1].set_title('LargeAF Performance')
        # axs[1].set_ylim(0, 1)
        # axs[1].set_ylabel('Response Left')
        # axs[1].set_xlabel('% Blue')
        # axs[1].scatter(self.summary[self.summary.attn_size == 'l']['diff'],\
        #                self.summary[self.summary.attn_size == 'l']['resp_blue'])
        # fig.savefig(f'./logs/{self.folder}_Logs/performance.png',dpi=300)

        # plot sigmoid and print 20% and 80% values

        fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(f'Attention Condition: {self.attn.upper()}', fontsize=14)

        for i,at in enumerate(['s', 'l']):
            xdata = self.summary[self.summary.attn_size == at]['diff']
            ydata = self.summary[self.summary.attn_size == at]['resp_blue']
            popt, pcov = curve_fit(sigmoid, xdata, ydata)
            val = (abs(0.5 - inv_sigmoid(.2, *popt)) + abs(0.5 - inv_sigmoid(.2, *popt))) / 2

            print(f'\nATTN: {at.upper()}\
                \nSigmoid mid-point: {inv_sigmoid(.5, *popt):.3f}\
                \nYes/No Values: {0.5 + val:.3f} , {0.5 - val:.3f}')

            x = np.linspace(0, 1, 20)
            y = sigmoid(x, *popt)

            
            axs[i].plot(xdata, ydata, 'o', label='data')
            axs[i].set_title(f'{at.upper()} Performance', fontsize=9)
            axs[i].plot(x, y, label='sigmoid')
            axs[i].set_ylim(0, 1)
            axs[i].set_ylabel('Response Blue')
            axs[i].set_xlabel('% Blue')
            plt.legend(loc='best')          


        fig2.savefig(f'./logs/{self.folder}_Logs/sigmoid.png',dpi=300)

        plt.show()

    def analyseYesNo(self):
        print(self.wd)
        fname = f'{self.wd}/logs/{self.folder}_Logs/*.tsv'
        sz = 'large_prop' if self.attn == 'l' else 'small_prop'
        baseline = 0.5
        duration = 1

        df = pd.read_table(glob.glob(fname)[0], keep_default_na=True)
        df = df.drop(
            df[(df.phase % 2 == 1) & (df.event_type == 'stim')].index.append(df[df.event_type == 'pulse'].index))
        df['duration'] = df['duration'].fillna(0)
        df['nr_frames'] = df['nr_frames'].fillna(0)
        df['end'] = df.onset + df.duration
        df['end_abs'] = df.onset_abs + df.duration

        stim_df = df[df.event_type == 'stim']
        switch_loc = np.diff(stim_df[sz], prepend=baseline) != 0
        switch_loc = stim_df[(switch_loc) & (stim_df[sz] != baseline)].index  # drop values where color_balance is 0.5
        responses = df.loc[df.response == 'space']

        tp = sum([(abs(i - responses.onset) < duration).any() \
                  for i in stim_df.loc[switch_loc].end])  # true positives
        fn = len(switch_loc) - tp  # false negatives (missed switches)
        fp = len(responses) - tp  # false positives (responded with no switch)
        tn = len(stim_df) - len(switch_loc) - fn  # true negative

        d, c = d_prime(tp, fn, fp, tn)

        print(f"D': {d:.3f}, C: {c:.3f}")

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
    elif fa_rate == 0:
        fa_rate = half_fa

    d_prime = Z(hit_rate) - Z(fa_rate)
    c = -(Z(hit_rate) + Z(fa_rate)) / 2
    #     print(f'Hit rate: \t {hit_rate} \nFalse Alarm rate: {fa_rate}')

    return d_prime, c
