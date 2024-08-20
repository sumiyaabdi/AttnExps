#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:04:44 2019

@author: sumiyaabdi
"""
import sys
import yaml
from session import PsychophysSession,PRFSession
from analyse import *
from datetime import datetime

datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    subject = sys.argv[1] # e.g. sub-001
    sess =  sys.argv[2] # e.g. 1
    run = sys.argv[3] # e.g. 0
    diff = sys.argv[4] # e.g. 'e' or 'h
    name = 'train'
    task='rsvp'
    attn=diff

    eyetrack = ''
    while eyetrack not in ('y','yes','n','no'):
        eyetrack = input('Eyetracking (y/n)?: ')
    
    output_str= subject+'_ses-'+sess+'_task-attn'+diff.upper()+'_run-'+run
    print(f'Output folder: {output_str}')
    
    output_str= subject+'_ses-'+sess+'_task-attn'+diff.upper()+'_run-'+run
    output_dir = f'./logs/{subject}/{output_str}_Logs'
    
    if os.path.exists(output_dir):
        print("Warning: output directory already exists. Renaming to avoid overwriting.")
        output_dir = output_dir + datetime.now().strftime('%Y%m%d%H%M%S')

    settings_file=f'expsettings/{name}settings_'+task+'.yml'
    with open(settings_file) as file:
        settings = yaml.safe_load(file)

    cond='easy' if diff=='e' else 'hard'
    print(output_str,'\n',output_dir,'\n',settings_file,'\n',cond)

    # if len(sys.argv) < 6:
    #     if (eyetrack == 'n') or (eyetrack == 'no'):
    #         ts = PsychophysSession(output_str=output_str,
    #                             output_dir=output_dir,
    #                             settings_file=settings_file,
    #                             difficulty=cond,
    #                             eyetracker_on=False)
    #     else:
    #         ts = PsychophysSession(output_str=output_str,
    #                             output_dir=output_dir,
    #                             settings_file=settings_file,
    #                             difficulty=cond,
    #                             ntrials=settings['rsvp']['ntrials'])
            
    if len(sys.argv) < 6:
        if (eyetrack == 'n') or (eyetrack == 'no'):
            ts = PRFSession(output_str=output_str,
                                output_dir=output_dir,
                                settings_file=settings_file,
                                difficulty=cond,
                                eyetracker_on=False)
        else:
            ts = PRFSession(output_str=output_str,
                                output_dir=output_dir,
                                settings_file=settings_file,
                                difficulty=cond,
                                eyetracker_on=True)
        ts.create_stimuli()
        ts.create_trials()
        ts.run()

    return output_str, task, diff,subject,name


if __name__ == '__main__':
    output_str, task,diff,subject,name = main()
    beh = AnalyseRSVP(output_str, task, diff,subject,name)
    beh.analyseYesNo()