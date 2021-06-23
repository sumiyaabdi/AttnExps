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
from utils import *

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

        self.resp_keys = self.settings['attn_task']['resp_keys']
        self.interp_values = self.settings['attn_task']['interp_values']

    def analyse2afc(self):

        f = glob.glob(f"{self.wd}/logs/{self.folder}_Logs/*.tsv")[0]

        sub,ses,task,run = [i.split('-')[-1] for i in self.folder.split('_')]
        sz = task[-1]
        task=task[:-1]
        resp_blue, resp_pink = [str(i).upper() for i in self.resp_keys]

        df = pd.read_table(f,keep_default_na=True)
        df = df[df.event_type == 'response']
        df.drop(['nr_frames','duration'],axis=1,inplace=True)
        df['response'] = df.response.astype(str).apply(lambda x:x.lower())
        # df['attn_size']=sz
        # df['corr_L']=np.nan
        # df['corr_S']=np.nan

        large_resp_blue = []
        small_resp_blue = []

        for prop in df['large_prop'].unique():
            large_resp_blue.append(sum(df.response[df['large_prop'] == prop] == resp_blue)\
                            / len(df.response[df['large_prop'] == prop] == resp_blue))
        for prop in df['small_prop'].unique():
            small_resp_blue.append(sum(df.response[df['small_prop'] == prop] == resp_blue)\
                            / len(df.response[df['small_prop'] == prop] == resp_blue))

        large_resp= (df['large_prop'].unique(),large_resp_blue)
        small_resp= (df['small_prop'].unique(),small_resp_blue)

        summary = [large_resp, small_resp]

        fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(f'Attention Condition: {self.attn.upper()}', fontsize=14)

        for i,at in enumerate(['l','s']):
            xdata = summary[i][0]
            ydata = summary[i][1]
            try:
                popt, pcov = curve_fit(sigmoid, xdata, ydata)
            except RuntimeError:
                print(f"\nError - {at.upper()} curve fit failed")
                continue
            val = (abs(0.5 - inv_sigmoid(self.interp_values[0], *popt)) + abs(0.5 - inv_sigmoid(self.interp_values[1], *popt))) / 2

            print(f'\nATTN: {at.upper()}\
                \nSigmoid mid-point: {inv_sigmoid(.5, *popt):.3f}\
                \nYes/No Values {self.interp_values}: {0.5 + val:.3f} , {0.5 - val:.3f}\
                \npopt,pcov {popt, pcov}\n')

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
        # resp = str(self.resp_keys[0])[0]

        df = pd.read_table(glob.glob(fname)[0], keep_default_na=True)
        df = df.drop(
            df[(df.phase % 2 == 1) & (df.event_type == 'stim')].index.append(df[df.event_type == 'pulse'].index))
        df['duration'] = df['duration'].fillna(0)
        df['nr_frames'] = df['nr_frames'].fillna(0)
        df['end'] = df.onset + df.duration
        df['end_abs'] = df.onset_abs + df.duration
        df['response'] = df.response.astype(str).apply(lambda x:x.lower())
        resp = df.response.unique()[1]

        stim_df = df[df.event_type == 'stim']
        switch_loc = np.diff(stim_df[sz], prepend=baseline) != 0
        switch_loc = stim_df[(switch_loc) & (stim_df[sz] != baseline)].index  # drop values where color_balance is 0.5
        responses = df.loc[df.response == resp]

        tp = sum([(abs(i - responses.onset) < duration).any() \
                  for i in stim_df.loc[switch_loc].end])  # true positives
        fn = len(switch_loc) - tp  # false negatives (missed switches)
        fp = len(responses) - tp  # false positives (responded with no switch)
        tn = len(stim_df) - len(switch_loc) - fn  # true negative

        d, c = d_prime(tp, fn, fp, tn)

        print(f"D': {d:.3f}, C: {c:.3f}")

# def sigmoid(x,x0,k):
#     y = np.array(1 / (1 + np.exp(-k*(x-x0))))
#     return y

# def weibull(x,x0,k,g,l):
#     y = g +(1-g -l)*sigmoid(x,k)
#     return y

# def inv_sigmoid(y,x0,k):
#     return x0 - (np.log((1/y)-1)/k)

# def d_prime(hits, misses, fas, crs):
#     """
#     calculate d' from hits(tp), misses(fn), false
#     alarms (fp), and correct rejections (tn)

#     returns: d_prime
#     """

#     half_hit = 0.5 / (hits + misses)
#     half_fa = 0.5 / (fas + crs)

#     hit_rate = hits / (hits + misses)
#     fa_rate = fas / (fas + crs)

#     # avoid d' infinity
#     if hit_rate == 1:
#         hit_rate = 1 - half_hit
#     elif hit_rate == 0:
#         hit_rate = half_hit

#     if fa_rate == 1:
#         fa_rate = 1 - half_fa
#     elif fa_rate == 0:
#         fa_rate = half_fa

#     d_prime = Z(hit_rate) - Z(fa_rate)
#     c = -(Z(hit_rate) + Z(fa_rate)) / 2
#     #     print(f'Hit rate: \t {hit_rate} \nFalse Alarm rate: {fa_rate}')

#     return d_prime, c
