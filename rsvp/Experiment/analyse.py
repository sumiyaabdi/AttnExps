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

    def __init__(self,folder,task,attn,subj,set,wd=None,stim_type=None,verbose=True):
        self.folder=folder
        self.task=task
        self.attn=attn
        self.wd = wd if wd else os.getcwd() 
        self.set = set
        self.subj = subj
        self.verbose=verbose
        self.diff = [i.split('-')[-1] for i in self.folder.split('_')][3]
        self.stim_type='letter'

        # settings_f = opj(self.wd, f'expsettings/{set}settings_{self.task}.yml')
        settings_f = opj(self.wd,f'logs/{self.subj}/{self.folder}_Logs/{self.folder}_expsettings.yml')
        if verbose:
            print('Settings file: ',settings_f)
        with open(settings_f) as file:
            self.settings = yaml.safe_load(file)

        self.resp_keys = self.settings['attn_task']['resp_keys']
        self.blue_key, self.pink_key = [str(i).lower() for i in self.resp_keys]

    def loadRunLog(self):
        f = glob.glob(f"{self.wd}/logs/{self.subj}/{self.folder}_Logs/*.tsv")[0]
        df = pd.read_table(f,keep_default_na=True)
        return df
    
    def combineRuns(self,folders):
        df = pd.DataFrame()
        folders.append(self.folder)
        print(folders)
        for folder in folders:
            f = glob.glob(f"{self.wd}/logs/{self.subj}/{folder}_Logs/*.tsv")[0]
            df = pd.concat((df,pd.read_table(f,keep_default_na=True)))
        return df
    
    def getPsychometric(self,func,inv_func,xdata,ydata,bounds=None):
        x = np.linspace(0, 1, 20)

        try:
            popt, _ = curve_fit(func, xdata, ydata,bounds)
        except (RuntimeError, UnboundLocalError):
            print(f"\nError - {self.diff.upper()} curve fit failed")

        easy_val = (abs(0.5 - inv_func(self.easyInterp[0], *popt)) + \
                    abs(0.5 - inv_func(self.easyInterp[1], *popt))) / 2
        hard_val = (abs(0.5 - inv_func(self.hardInterp[0], *popt)) + \
                    abs(0.5 - inv_func(self.hardInterp[1], *popt))) / 2
        
        if len(self.easyInterp) >= 4:
            hard_val2 = (abs(0.5 - inv_func(self.hardInterp[2], *popt)) + abs(0.5 - inv_func(self.hardInterp[3], *popt))) / 2
            easy_val2 = (abs(0.5 - inv_func(self.easyInterp[2], *popt)) + abs(0.5 - inv_func(self.easyInterp[3], *popt))) / 2

            print(f'\nATTN: {self.diff.upper()}\
                \n{func.__name__.upper()}: {inv_func(.5, *popt):.3f}\
                \nEasy Values {self.easyInterp}: {0.5 + easy_val:.3f} , {0.5 - easy_val:.3f}, {0.5 + easy_val2:.3f} , {0.5 - easy_val2:.3f}\
                \nHard Values {self.hardInterp}: {0.5 + hard_val:.3f} , {0.5 - hard_val:.3f}, {0.5 + hard_val2:.3f} , {0.5 - hard_val2:.3f}')
        else:
            print(f'\nATTN: {self.diff.upper()}\
            \n{func.__name__.upper()}: {inv_func(.5, *popt):.3f}\
            \nEasy Values {self.easyInterp}: {0.5 + easy_val:.3f} , {0.5 - easy_val:.3f}\
            \nHard Values {self.hardInterp}: {0.5 + hard_val:.3f} , {0.5 - hard_val:.3f}')

        y = func(x, *popt)
        return y, popt


    def plotPsychometric(self,func='sigmoid',folders=None):
        if folders == None:
            df = self.loadRunLog()
        else:
            df = self.combineRuns(folders)

        df = df[df.event_type == 'response']
        df['response'] = df.response.astype(str).apply(lambda x:x.lower())

        resp_blue,resp_pink = [],[]

        for prop in df['small_prop'].unique():
            resp_blue.append(sum(df.response[df['small_prop'] == prop] == self.blue_key)\
                            / len(df.response[df['small_prop'] == prop] == self.blue_key))
            resp_pink.append(sum(df.response[df['small_prop'] == prop] == self.pink_key)\
                            / len(df.response[df['small_prop'] == prop] == self.pink_key))

        summary = (df['small_prop'].unique(),resp_blue)

        xdata = summary[0]
        ydata = summary[1]
        
        x = np.linspace(0, 1, 20)

        if func == 'sigmoid':
            y,popt=self.getPsychometric(sigmoid,inv_sigmoid,xdata,ydata)

        elif func == 'fix-mid':
            y,popt=self.getPsychometric(sigmoid_fixmid,inv_sigmoid_fix,xdata,ydata)

        elif func == 'weibull':
            y,popt=self.getPsychometric(weibull,inv_weibull,xdata,ydata)

        fig2, axs = plt.subplots(1, 1, figsize=(8, 6))
        fig2.suptitle(f'Attention Condition: {self.diff.upper()}', fontsize=14)
        axs.plot(xdata, ydata, 'o', label='data')
        # axs.plot(xdata_pink, ydata_pink, 'o',c='pink')

        # axs.set_title(f'Difficulty {diff.upper()} Performance', fontsize=9)
        axs.plot(x, y,c='blue',label='fit')
        # axs.plot(x, y_pink, c='pink')
        axs.set_ylim(0, 1)
        axs.set_ylabel('Response Blue')
        axs.set_xlabel('% Blue')
        plt.legend(loc='best')          

        fig2.savefig(f'./logs/{self.subj}/{self.folder}_Logs/sigmoid.png',dpi=300)

        plt.show()

    def printPerformance(self,folders=None):

        if folders:
            df = self.combineRuns(folders)
        else:
            df = self.loadRunLog()
        df['response'] = df.response.astype(str).apply(lambda x:x.lower())
        df.drop(df[(pd.isna(df[self.stim_type]) & (df.event_type == 'stim'))].index,inplace=True) # remove baseline trials
        df.drop(df[(df.phase == 1) & (df.event_type == 'stim')].index,inplace=True) # keep only 1 stim phase for simplicity

        corr_count = 0
        miss_trial = 0
        mult_trial = 0
        incorr_count = 0
        corr_rts = []
        incorr_rts = []
        df['performance'] = np.nan

        for trial in df.trial_nr.unique():
            # if there is no response
            if not any(df[df.trial_nr == trial].event_type == 'response'):
                df.loc[df.trial_nr == trial,'performance']=np.nan
                miss_trial+=1
                incorr_count+=1
                # print('miss')
                continue
            # if there are more than 1 responses
            elif len(df[(df.trial_nr == trial ) & (df.event_type == 'response')]) > 1:
                df.loc[df.trial_nr == trial,'performance']=np.nan
                mult_trial+=1
                incorr_count+=1
                # print('more than 1 response')
                continue
            # if stim more pink
            elif all(df[(df.trial_nr == trial) & (df.event_type == 'response')].small_prop < 0.5):
                if all(df[(df.trial_nr == trial) & (df.event_type == 'response')].response == self.pink_key):
                    df.loc[df.trial_nr == trial,'performance']=1
                    corr_count+=1
                    corr_rts.append(df[(df.trial_nr == trial)].onset.diff().unique()[1])
                    # print('correct, pink')
                elif all(df[(df.trial_nr == trial) & (df.event_type == 'response')].response == self.blue_key):
                    df.loc[df.trial_nr == trial,'performance']=0
                    incorr_count+=1
                    incorr_rts.append(df[(df.trial_nr == trial)].onset.diff().unique()[1])
                    # print('incorrect, pink')
            # if stim more blue
            elif all(df[(df.trial_nr == trial) & (df.event_type == 'response')].small_prop > 0.5):
                if all(df[(df.trial_nr == trial) & (df.event_type == 'response')].response == self.blue_key):
                    df.loc[df.trial_nr == trial,'performance']=1
                    corr_count+=1
                    corr_rts.append(df[(df.trial_nr == trial)].onset.diff().unique()[1])
                    # print('correct, blue')
                elif all(df[(df.trial_nr == trial) & (df.event_type == 'response')].response == self.pink_key):
                    df.loc[df.trial_nr == trial,'performance']=0
                    incorr_count+=1
                    incorr_rts.append(df[(df.trial_nr == trial)].onset.diff().unique()[1])
                    # print('incorrect, blue')
            else:
                df[df.trial_nr == trial]['performance']=np.nan

        self.df = df
        print(df.columns)
        print(f'Proportions: {df.small_prop.unique()} \n \
        N Trials: {len(df.trial_nr.unique())} \n \
        Corr: {100*corr_count/len(df.trial_nr.unique()):.1f}%  Avg RT: {np.mean(corr_rts):.3f} \n \
        Incorr: {100*incorr_count/len(df.trial_nr.unique()):.1f}%  Avg RT: {np.mean(incorr_rts):.3f} \n \
        # Miss trials: {miss_trial} ({100*miss_trial/len(df.trial_nr.unique()):.1f}%) \n \
        # Multiple response trials: {mult_trial} ({100*mult_trial/len(df.trial_nr.unique()):.1f}%)')

    # def plotPropCorrect(self):
    #     for prop in self.df.small_prop.unique()


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

            print(f'\nDiff: {at.upper()}\
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

class AnalyseRSVP(AnalyseRun):

    def __init__(self,folder,task,attn,subj,set,wd=None,verbose=True):
        super().__init__(folder,task,attn,subj,set,wd,verbose)
        
        # settings_f = opj(self.wd, f'expsettings/{set}settings_{self.task}.yml')
        # if verbose:
        #     print('Settings file: ',settings_f)
        # with open(settings_f) as file:
        #     self.settings = yaml.safe_load(file)

        self.stim_type='letter'
        
        self.h_target_features={feat:self.settings['rsvp'][f'h_target_{feat}'] for feat in self.settings['rsvp']['h_target_features']}
        self.h_letter=self.h_target_features['letter']

        self.e_letter=self.settings['rsvp']['e_target']


    def analyseYesNo(self,resp=None,resp_time=None,fname=None):

        if not fname:
            fname = f'{self.wd}/logs/{self.subj}/{self.folder}_Logs/{self.folder}_events.tsv'
        cond = 'h' if self.attn == 'h' else 'e'
        
        duration = self.settings['attn_task']['resp_time'] if resp_time == None else resp_time
        resp = resp if resp else str(self.resp_keys[0])[0]
        assert os.path.exists(fname), f'file does not exist {fname}'

        df = self.loadRunLog() # pd.read_table(fname, keep_default_na=True)

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

        # correct for accidental press of other key
        if resp not in df.response.unique():
            df.loc[df.response != 'nan','response'] = resp

        stim_df = df[df.event_type == 'stim']
        prop_values=self.e_letter if cond == 'e' else self.h_letter

        targdf={}
        targdf['e']=df[(df['event_type']=='stim')& ((df['letter']==self.e_letter[0]) | (df['letter']==self.e_letter[1]))]
        targdf['h']=pd.concat((df[(df['event_type']=='stim')&(df['letter']==self.h_target_features['letter'][0])&(df['ori']==self.h_target_features['ori'][0])],\
                    df[(df['event_type']=='stim')&(df['letter']==self.h_target_features['letter'][1])&(df['ori']==self.h_target_features['ori'][1])]))

        if self.verbose:
            print(f'\nAttention {self.attn.upper()}\Target(s): {prop_values}\nResponse Keys: {df.response.unique()}\n')
        
        self.targdf=targdf
        self.df=df
        for diff in ['h', 'e']:
            switch_loc=targdf[diff].index
            # switch_loc = np.diff(stim_df[sz], prepend=baseline) != 0
            # switch_loc = stim_df[(switch_loc) & (stim_df[sz] != baseline) & (stim_df[sz].notna())].index  # drop values where color_balance is 0.5
            
            responses = df.loc[df.response == resp]

            tp = sum([(abs(i - responses.onset) < duration).any() \
                    for i in stim_df.loc[switch_loc].end])  # true positives
            fn = len(switch_loc) - tp  # false negatives (missed switches)
            fp = len(responses) - tp  # false positives (responded with no switch)
            tn = len(stim_df) - len(switch_loc) - fn  # true negative

            d, c = d_prime(tp, fn, fp, tn)

            rts = [min(abs(responses.onset - i)) for i in df.loc[switch_loc].onset]
            rts = [r for r in rts if r < 2]            

            # if diff == self.attn:
            self.prop=prop_values
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

                
                if diff == self.attn:
                    self.ontask=True
                    print('On task results')
                else:
                    self.ontask=False
                    print('Off task results')
                print(f"{diff} D': {d:.3f}, C: {c:.3f}\n")
                print(f"{len(self.switch_loc)} expected responses\
                        \n{len(self.responses)} actual subject responses\
                        \n{self.tp} hits (within {self.settings['attn_task']['resp_time']}s)\
                        \n{self.fn} misses\
                        \n{self.fp} false alarms\
                        \n{self.tn} correct rejections\
                        \nAverage RT: {self.rt:.3f}s\n")
    

class AnalyseSession():
    def __init__(self,subject,ses='ses-1',wd=None,verbose=True):
        self.subject=subject
        self.wd = wd if wd else os.getcwd() 
        self.verbose=verbose

        self.runfolders=glob.glob(f"{self.wd}/logs/{self.subject}/{self.subject}_{ses}_task-*_Logs")

        for runfolder in self.runfolders:
            folder=runfolder.split('/')[-1]
            task=folder.split('_')[2].split('-')[-1]
            attn=folder.split('_')[2][-1].lower()
            name='exp'
            
            self.run=AnalyseRSVP(folder,task,attn,self.subject,name)
                                        

        
        
