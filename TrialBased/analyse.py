import yaml
import pickle
import pandas as pd
from psychopy import data
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

class AnalyseTrialBased():
    """Runs after the experiment is completed. Returns a summary of the results.
    """
    
    def __init__(self,output_str,wd=None,verbose=True):
        self.output_str=output_str
        self.subj=output_str.split('_')[0]
        self.wd = wd if wd else os.getcwd() 
        self.verbose=verbose
        self.logFolder = f'./logs/{self.subj}/{output_str}_Logs'
        
        with open(opj(self.logFolder,self.output_str+'_expsettings.yml')) as file:
            self.settings = yaml.safe_load(file)


    def load_stairs(self):
        self.behTypes=self.settings['trial_types']['response_types']
        fns = [opj(self.logFolder,self.output_str+f'_{beh}.npy') for beh in self.behTypes]
        self.stair_data={}
        for beh,fn in zip(self.behTypes,fns):
            with open(fn, 'rb') as f:
                self.stair_data[beh] = pickle.load(f)
        self.stair_data
    
    def plot_stairs(self):
        colors=['tab:blue','tab:purple','tab:orange','tab:red']
        threshVal=0.8
        expectedMin=0
        maxIntensity=None

        fig,ax=plt.subplots(1,2,figsize=(12,6))

        for beh,color in zip(self.behTypes,colors):
            if not maxIntensity:
                maxIntensity=np.max(self.stair_data[beh].intensities)
            if maxIntensity < np.max(self.stair_data[beh].intensities):
                maxIntensity=np.max(self.stair_data[beh].intensities)
            ax[0].scatter(np.where(np.asarray(self.stair_data[beh].data,dtype=int) == 0)[0],
                    0.5-np.asarray(self.stair_data[beh].intensities)[np.asarray(self.stair_data[beh].data,dtype=int) == 0], 
                    marker='x',c='black')
            ax[0].scatter(np.where(np.asarray(self.stair_data[beh].data,dtype=int) == 1)[0],
                    0.5-np.asarray(self.stair_data[beh].intensities)[np.asarray(self.stair_data[beh].data,dtype=int) == 1], 
                    marker='^',c='black')
            ax[0].plot(0.5-np.asarray(self.stair_data[beh].intensities),label=beh,color=color)

            
            for val in np.unique(self.stair_data[beh].intensities):
                ids=np.where(self.stair_data[beh].intensities == val)[0]
                ax[1].plot(val,np.asarray(self.stair_data[beh].data)[ids].sum()/len(ids),color=color,markerfacecolor='None',marker='o',ms=(len(ids)+1)*2)
            
            i, r, n = data.functionFromStaircase(self.stair_data[beh].intensities, self.stair_data[beh].data, bins='unique')
            n=np.asarray(n)
            try:
                fit=data.FitWeibull(i, r, expectedMin=expectedMin,sems=1.0 / n)
            except RuntimeError:
                print(f'Error fitting {beh}')
                continue
            smoothInt = np.arange(min(i), max(i), 0.001)
            smoothResp = fit.eval(smoothInt)
            thresh = fit.inverse(threshVal)
            print(beh, ': ', thresh)

            ax[1].plot(smoothInt, smoothResp, color=color,label=beh)
            ax[1].plot([thresh, thresh], [0, threshVal], '--',color=color)  # vertical dashed line
            ax[1].plot([0, thresh], [threshVal, threshVal], '--',color=color)  # horizontal dashed line
            ax[1].set_ylim([-0.05, 1.05])
            ax[1].set_xlim([0, maxIntensity])

        # fig.suptitle('Staircase data',fontsize=22)

        ax[0].set_title('Intensity per trial',fontsize=18)
        ax[0].legend()
        ax[0].set_xlabel('Trial',fontsize=15)
        ax[0].set_ylabel('Intensity (distance from prop 0.5)',fontsize=15)

        ax[1].set_title('Psychometric Functions',fontsize=18)
        ax[1].legend(bbox_to_anchor=(1, 0), loc='lower right')
        ax[1].set_xlabel('Intensity',fontsize=15)
        ax[1].set_ylabel('Proportion correct',fontsize=15)

        fig.savefig(opj(self.logFolder,'staircase.png'),dpi=300)
        plt.show()


# class AnalyseRun():

#     def __init__(self,folder,task,attn,subj,set,wd=None,verbose=True):
#         self.folder=folder
#         self.task=task
#         self.attn=attn
#         self.wd = wd if wd else os.getcwd() 
#         self.set = set
#         self.subj = subj
#         self.verbose=verbose

#         settings_f = opj(self.wd, f'expsettings/{set}settings_{self.task}.yml')
#         if verbose:
#             print('Settings file: ',settings_f)
#         with open(settings_f) as file:
#             self.settings = yaml.safe_load(file)

#         self.resp_keys = self.settings['attn_task']['resp_keys']
#         self.interp_values = self.settings['attn_task']['interp_values']

#     def analyse2afc(self):

#         sub,ses,task,run = [i.split('-')[-1] for i in self.folder.split('_')]
#         f = glob.glob(f"{self.wd}/logs/{self.subj}/{self.folder}_Logs/*.tsv")[0]


#         sz = task[-1]
#         task=task[:-1]

#         df = pd.read_table(f,keep_default_na=True)
#         df = df[df.event_type == 'response']
#         # df.drop(['nr_frames','duration'],axis=1,inplace=True)
#         df['response'] = df.response.astype(str).apply(lambda x:x.lower())

#         for ix,resp in enumerate(self.resp_keys):
#             print(resp)
#             if str(resp) not in df['response'].unique(): # edge-case for differently labelled responses '1.0' in psychophys room
#                 self.resp_keys[ix] = str(resp)[0]

#         blue_key, pink_key = [str(i).lower() for i in self.resp_keys]

                
#         fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
#         fig2.suptitle(f'Attention Condition: {self.attn.upper()}', fontsize=14)

#         for i,at in enumerate(['l','s']):
#             cond = 'large' if at == '' else 'small'
    
#             resp_blue = []

#             for prop in df[f'{cond}_balance'].unique():
#                 print()
#                 resp_blue.append(sum(df.response[df[f'{cond}_balance'] == prop] == blue_key)\
#                                 / len(df.response[df[f'{cond}_balance'] == prop] == blue_key))

#             xdata = df[cond+'_balance'].unique()
#             ydata = resp_blue
#             try:
#                 popt, pcov = curve_fit(sigmoid, xdata, ydata)
#             except RuntimeError:
#                 print(f"\nError - {at.upper()} curve fit failed")
#                 continue
#             val = (abs(0.5 - inv_sigmoid(self.interp_values[0], *popt)) + abs(0.5 - inv_sigmoid(self.interp_values[1], *popt))) / 2

#             print(f'\nATTN: {at.upper()}\
#                 \nSigmoid mid-point: {inv_sigmoid(.5, *popt):.3f}\
#                 \nYes/No Values {self.interp_values}: {0.5 + val:.3f} , {0.5 - val:.3f}\
#                 \npopt,pcov {popt, pcov}\n')

#             x = np.linspace(0, 1, 20)
#             y = sigmoid(x, *popt)
            
#             axs[i].plot(xdata, ydata, 'o', label='data')
#             axs[i].set_title(f'{at.upper()} Performance', fontsize=9)
#             axs[i].plot(x, y, label='sigmoid')
#             axs[i].set_ylim(0, 1)
#             axs[i].set_ylabel('Response Blue')
#             axs[i].set_xlabel('% Blue')
#             plt.legend(loc='best')          

#         fig2.savefig(f'./logs/{self.subj}/{self.folder}_Logs/sigmoid.png',dpi=300)

#         plt.show()


#     def analyseYesNo(self,resp=None,fname=None):
#         if not fname:
#             fname = f'{self.wd}/logs/{self.subj}/{self.folder}_Logs/{self.folder}_events.tsv'
#         cond = 'large_balance' if self.attn == 'l' else 'small_balance'
#         baseline = 0.5
#         duration = self.settings['attn_task']['resp_time']
#         resp = resp if resp else str(self.resp_keys[0])[0]
#         assert os.path.exists(fname), f'file does not exist {fname}'

#         df = pd.read_table(fname, keep_default_na=True)

#         try:
#             df['duration'] = df['duration'].fillna(0)
#         except KeyError:
#             df.loc[df.event_type == 'stim','duration'] = np.diff(df.loc[df.event_type == 'stim','onset'],append=np.nan)
    
#         df = df.drop(df[(df.phase % 2 == 1) & (df.event_type == 'stim')].index.append(df[df.event_type == 'pulse'].index))
#         df.drop(df[(df.duration > 1)].index,inplace=True)
#         df.drop(df[pd.isna(df.duration) & (df.event_type != 'response')].index,inplace=True)

#         df['nr_frames'] = df['nr_frames'].fillna(0)
#         df['end'] = df.onset + df.duration
#         df['end_abs'] = df.onset_abs + df.duration
#         df['response'] = df.response.astype(str).apply(lambda x:x.lower())

#         if resp not in df.response.unique():
#             df.loc[df.response != 'nan','response'] = resp

#         stim_df = df[df.event_type == 'stim']

#         prop_values=[prop for prop in df[cond].unique() if not pd.isna(prop)]
#         prop_values=[p for p in prop_values if p != 0.5]

#         if self.verbose:
#             print(f'\nAttention {self.attn.upper()}\nProportions: {prop_values}\nResponse Keys: {df.response.unique()}\n')

#         for sz in ['small_balance', 'large_balance']:
#             # prop_values=[]
#             switch_loc = np.diff(stim_df[sz], prepend=baseline) != 0
#             switch_loc = stim_df[(switch_loc) & (stim_df[sz] != baseline) & (stim_df[sz].notna())].index  # drop values where color_balance is 0.5
            
#             responses = df.loc[df.response == resp]

#             tp = sum([(abs(i - responses.onset) < duration).any() \
#                     for i in stim_df.loc[switch_loc].end])  # true positives
#             fn = len(switch_loc) - tp  # false negatives (missed switches)
#             fp = len(responses) - tp  # false positives (responded with no switch)
#             tn = len(stim_df) - len(switch_loc) - fn  # true negative

#             d, c = d_prime(tp, fn, fp, tn)

#             rts = [min(abs(responses.onset - i)) for i in df.loc[switch_loc].onset]
#             rts = [r for r in rts if r < 2]            

#             if sz[0] == self.attn:
#                 self.prop=prop_values
#                 self.fname = glob.glob(fname)[0]
#                 self.d = d
#                 self.c = c
#                 self.tp = tp
#                 self.fn = fn
#                 self.fp = fp
#                 self.tn = tn
#                 self.switch_loc = switch_loc
#                 self.responses = responses
#                 self.df = df
#                 self.rt = np.mean(rts)

#             if self.verbose:
#                 print(f"{sz.split('_')[0]} D': {d:.3f}, C: {c:.3f}\n")
                
#                 if sz[0] == self.attn:
#                     print(f"{len(self.switch_loc)} expected responses\
#                           \n{len(self.responses)} actual subject responses\
#                           \n{self.tp} hits (within {self.settings['attn_task']['resp_time']}s)\
#                           \n{self.fn} misses\
#                           \n{self.fp} false alarms\
#                           \nAverage RT: {self.rt:.3f}s\n")