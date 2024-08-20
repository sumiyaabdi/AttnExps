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
from .utils import *

Z = norm.ppf

##  UNDER CONSTRUCTION ##

class AnalyseTrialSession():
    def __init__(self, subject, session=1):
        self.subject = subject
        self.session = session
        self.logFolder = f'./logs/{self.subject}/'
        self._allevents = None
        self._settings = None
        self._behTypes = None
        self._stair_data = None
        self._runs = None
    
    @property
    def settings(self):
        if self._settings is None:
            with open('expsettings/settings.yml') as file:
                self._settings = yaml.safe_load(file)
        return self._settings
    
    @property
    def runs(self):
        if self._runs is None:
            run_fns=[fn for fn in os.listdir(opj(self.logFolder)) if f'{self.subject}_ses-{self.session}' in fn]
            for fn in run_fns:
                run_num=fn.split('_')[-1]
                self._runs[run_num]=AnalyseTrialRun(fn)
        return self._runs

class AnalyseTrialRun():
    """Runs after the experiment is completed. Returns a summary of the results.
    """
    
    def __init__(self,output_str,wd=None,verbose=True):
        self.output_str=output_str
        self.subj=output_str.split('_')[0]
        self.wd = wd if wd else os.getcwd() 
        self.verbose=verbose
        self.logFolder = f'./logs/{self.subj}/{output_str}_Logs'
        self._events = None
        self._stair_data = None
        
        with open(opj(self.logFolder,self.output_str+'_expsettings.yml')) as file:
            self.settings = yaml.safe_load(file)
        
        self.behTypes=self.settings['trial_types']['response_types']

    @property
    def events(self):      
        if self._events is None:
            self._events=pd.read_csv(opj(self.logFolder,self.output_str+'_events.tsv'),sep='\t')
        return self._events

    @property
    def stair_data(self):
        if self._stair_data is None:
            fns = [opj(self.logFolder,self.output_str+f'_{beh}.npy') for beh in self.behTypes]
            self._stair_data={}
            for beh,fn in zip(self.behTypes,fns):
                try:
                    with open(fn, 'rb') as f:
                        self.stair_data[beh] = pickle.load(f)
                except FileNotFoundError:
                    print(f'No staircase data found for {beh}')
                    continue
        return self._stair_data

    def plot_stairs(self):
        colors=['tab:blue','tab:purple','tab:orange','tab:red', 'tab:green']
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