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
    subject = sys.argv[1]
    sess =  sys.argv[2]
    # 5 conditions: PRF2R, PRF1R, PRF1S, PRF4R, PRF4F 
    #(2 squares Regular speed, 1 square Regular, 1 square Slow, 4 square Regular, 4 square Fast)
    # task = sys.argv[3]
    #in the full experiment we would do 3 runs
    run = sys.argv[3]

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
        if (eyetrack == 'n') or (eyetrack == 'n'):
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
        if (eyetrack == 'n') or (eyetrack == 'n'):
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
    ts = AnalyseRun(output_str, task, attn)

    if task == '2afc':
        ts.analyse2afc()
        ts.plot2afc()
    elif task == 'yesno':
        ts.analyseYesNo()
