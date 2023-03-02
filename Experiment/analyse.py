import yaml
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
pd.options.mode.chained_assignment = None
# import warnings
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
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

    def __init__(self,folder,task,attn,subj,set,wd=None,verbose=True):
        self.folder=folder
        self.task=task
        self.attn=attn
        self.wd = wd if wd else os.getcwd() 
        self.set = set
        self.subj = subj
        self.verbose=verbose

        settings_f = opj(self.wd, f'expsettings/{set}settings_{self.task}.yml')
        if verbose:
            print('Settings file: ',settings_f)
        with open(settings_f) as file:
            self.settings = yaml.safe_load(file)

        self.resp_keys = self.settings['attn_task']['resp_keys']
        self.interp_values = self.settings['attn_task']['interp_values']

    def analyse2afc(self):

        sub,ses,task,run = [i.split('-')[-1] for i in self.folder.split('_')]
        f = glob.glob(f"{self.wd}/logs/{self.subj}/{self.folder}_Logs/*.tsv")[0]


        sz = task[-1]
        task=task[:-1]
        resp_blue, resp_pink = [str(i).lower() for i in self.resp_keys]

        df = pd.read_table(f,keep_default_na=True)
        df = df[df.event_type == 'response']
        # df.drop(['nr_frames','duration'],axis=1,inplace=True)
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

        fig2.savefig(f'./logs/{self.subj}/{self.folder}_Logs/sigmoid.png',dpi=300)

        plt.show()


    def analyseYesNo(self,resp=None):
        fname = f'{self.wd}/logs/{self.subj}/{self.folder}_Logs/{self.folder}_events.tsv'
        cond = 'large_prop' if self.attn == 'l' else 'small_prop'
        baseline = 0.5
        duration = self.settings['attn_task']['resp_time']
        resp = resp if resp else str(self.resp_keys[0])[0]
        assert os.path.exists(fname), f'file does not exist {fname}'

        df = pd.read_table(fname, keep_default_na=True)

        try:
            df['duration'] = df['duration'].fillna(0)
        except KeyError:
            df.loc[df.event_type == 'stim','duration'] = np.diff(df.loc[df.event_type == 'stim','onset'],append=np.nan)
    
        df = df.drop(df[(df.phase % 2 == 1) & (df.event_type == 'stim')].index.append(df[df.event_type == 'pulse'].index))
        df.drop(df[(df.duration > 1)].index,inplace=True)
        df.drop(df[pd.isna(df.duration) & (df.event_type != 'response')].index,inplace=True)

        df['nr_frames'] = df['nr_frames'].fillna(0)
        df['end'] = df.onset + df.duration
        df['end_abs'] = df.onset_abs + df.duration
        df['response'] = df.response.astype(str).apply(lambda x:x.lower())

        if resp not in df.response.unique():
            df.loc[df.response != 'nan','response'] = resp

        stim_df = df[df.event_type == 'stim']

        prop_values=[prop for prop in df[cond].unique() if not pd.isna(prop)]
        prop_values=[p for p in prop_values if p != 0.5]

        if self.verbose:
            print(f'\nAttention {self.attn.upper()}\nProportions: {prop_values}\nResponse Keys: {df.response.unique()}\n')

        for sz in ['small_prop', 'large_prop']:
            prop_values=[]
            switch_loc = np.diff(stim_df[sz], prepend=baseline) != 0
            switch_loc = stim_df[(switch_loc) & (stim_df[sz] != baseline) & (stim_df[sz].notna())].index  # drop values where color_balance is 0.5
            
            responses = df.loc[df.response == resp]

            tp = sum([(abs(i - responses.onset) < duration).any() \
                    for i in stim_df.loc[switch_loc].end])  # true positives
            fn = len(switch_loc) - tp  # false negatives (missed switches)
            fp = len(responses) - tp  # false positives (responded with no switch)
            tn = len(stim_df) - len(switch_loc) - fn  # true negative

            d, c = d_prime(tp, fn, fp, tn)

            rts = [min(abs(responses.onset - i)) for i in df.loc[switch_loc].onset]
            rts = [r for r in rts if r < 2]            

            if sz[0] == self.attn:
                self.fname = glob.glob(fname)[0]
                self.d = d
                self.c = c
                self.tp = tp
                self.fn = fn
                self.fp = fp
                self.tn = tn
                self.switch_loc = switch_loc
                self.responses = responses
                self.df = df
                self.rt = np.mean(rts)

            if self.verbose:
                print(f"{sz.split('_')[0]} D': {d:.3f}, C: {c:.3f}\n")
                
                if sz[0] == self.attn:
                    print(f"{len(self.switch_loc)} expected responses\
                          \n{len(self.responses)} actual subject responses\
                          \n{self.tp} hits (within {self.settings['attn_task']['resp_time']}s)\
                          \n{self.fn} misses\
                          \n{self.fp} false alarms\
                          \nAverage RT: {self.rt:.3f}s\n")