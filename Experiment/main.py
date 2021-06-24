#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:04:44 2019

@author: marcoaqil
"""
import sys
from session import PRFSession, PsychophysSession
from analyse import *
from datetime import datetime



datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    subject = sys.argv[1] # e.g. sub-001
    sess =  sys.argv[2] # e.g. 1
    run = sys.argv[3] # e.g. 0

    task = ''
    while task not in ('2afc', 'yesno'):
        task = input("Which attention task ['2afc' / 'yesno']?: ")

    attn = ''
    while attn not in ('s','l'):
        attn = input('Which attention size [small (s) / large (l)]?: ')

    eyetrack = ''
    while eyetrack not in ('y','yes','n','no'):
        eyetrack = input('Eyetracking (y/n)?: ')
    
    output_str= subject+'_ses-'+sess+'_task-'+task+attn.upper()+'_run-'+run
    print(f'Output folder: {output_str}')
    
    output_dir = './logs/'+output_str+'_Logs'
    
    if os.path.exists(output_dir):
        print("Warning: output directory already exists. Renaming to avoid overwriting.")
        output_dir = output_dir + datetime.now().strftime('%Y%m%d%H%M%S')

    settings_file='expsettings/expsettings_'+task+'.yml'

    if task == 'yesno':
        if (eyetrack == 'n') or (eyetrack == 'no'):
            ts = PRFSession(output_str=output_str,
                            output_dir=output_dir,
                            settings_file=settings_file,
                            eyetracker_on=False)
        else:
            ts = PRFSession(output_str=output_str,
                            output_dir=output_dir,
                            settings_file=settings_file)
        ts.create_stimuli()
        ts.create_trials()
        ts.run()

    elif task == '2afc':
        if (eyetrack == 'n') or (eyetrack == 'no'):
            ts = PsychophysSession(output_str=output_str,
                                   output_dir=output_dir,
                                   settings_file=settings_file,
                                   eyetracker_on=False)
        else:
            ts = PsychophysSession(output_str=output_str,
                                   output_dir=output_dir,
                                   settings_file=settings_file)
        ts.create_stimuli()
        ts.create_trials()
        ts.run()

    return output_str, task, attn


if __name__ == '__main__':
    output_str, task, attn = main()
    beh = AnalyseRun(output_str, task, attn)

    if task == '2afc':
        beh.analyse2afc()
        # beh.plot2afc()
    elif task == 'yesno':
        beh.analyseYesNo()
