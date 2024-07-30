#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sumiyaabid
"""
import sys
import yaml
from session import AttnSession
from analyse import *
from datetime import datetime
import psychopy

print(psychopy.__version__)

datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    subject = sys.argv[1] # e.g. sub-001
    sess =  sys.argv[2] # e.g. 1
    run = sys.argv[3] # e.g. 0
    name = 'train'

    task = '2afc'
    # while task not in ('2afc', 'yesno'):
    #     task = input("Which attention task ['2afc' / 'yesno']?: ")

    attn = 's'
    # while attn not in ('s','l'):
    #     attn = input('Which attention size [small (s) / large (l)]?: ')

    eyetrack = 'n'
    # while eyetrack not in ('y','yes','n','no'):
    #     eyetrack = input('Eyetracking (y/n)?: ')
    
    output_str= subject+'_ses-'+sess+'_task-'+task+attn.upper()+'_run-'+run
    print(f'Output folder: {output_str}')
    
    output_dir = f'./logs/{subject}/{output_str}_Logs'
    
    if os.path.exists(output_dir):
        print("Warning: output directory already exists. Renaming to avoid overwriting.")
        output_dir = output_dir + datetime.now().strftime('%Y%m%d%H%M%S')

    settings_file=f'expsettings/settings.yml'
    with open(settings_file) as file:
        settings = yaml.safe_load(file)

    # use startVal from last run if possible
    last_outstr=output_str[:-1]+str(int(output_str[-1])-1)
    last_outdir=f'./logs/{subject}/{last_outstr}_Logs'
    try:
        last_tb=AnalyseTrialBased(last_outstr)
        last_tb.load_stairs()
        startVal=np.asarray([last_tb.stair_data[beh].intensities[-1] for beh in last_tb.behTypes]).mean()
        print(f'Using last run to start staircase, startVal = {startVal}')
        settings['staircase']['startVal']=startVal
    except FileNotFoundError:
        print('No previous staircase found. Starting from scratch.')

    ts = AttnSession(output_str=output_str,
                        output_dir=output_dir,
                        settings_file=settings_file,
                        eyetracker_on=False)

    ts.create_stimuli()
    ts.create_trials()
    ts.create_staircase()
    ts.run()

    return output_str 


if __name__ == '__main__':
    output_str = main()
    tb=AnalyseTrialBased(output_str)
    tb.load_stairs()
    tb.plot_stairs()

    # if task == '2afc':
    #     beh.analyse2afc()
    #     # beh.plot2afc()
    # elif task == 'yesno':
    #     beh.analyseYesNo()
