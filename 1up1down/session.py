#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sumiyaabdi
"""

import numpy as np
import pickle
rng=np.random.default_rng()
# np.random.seed(2024)
import os
import copy
import sys
sys.path.append('../../exptools2')

from psychopy import visual, tools, data
from psychopy.visual import filters

from exptools2.core import Session, PylinkEyetrackerSession
from trial import AttnTrial, BlankTrial
from stim import AttSizeStim, FixationStim,cross_fixation,HemiFieldStim,cueStim
from utils import create_stim_list, get_stim_nr,psyc_stim_list


opj = os.path.join

class AttnSession(PylinkEyetrackerSession):
    def __init__(self, output_str, 
                 output_dir, 
                 settings_file, 
                 eyetracker_on=True):
        """Session class contains all the information and methods to run the experiment.

        Args:
            output_str (str): Log folder substring
            output_dir (str): Folder where logs are saved
            settings_file (str): path/to/settings.yml
            eyetracker_on (bool, optional): Initiates eye calibration and tracking. Defaults to True.
        """

        super().__init__(output_str=output_str, 
                         output_dir=output_dir, 
                         settings_file=settings_file, 
                         eyetracker_on=eyetracker_on)

        n_conds=np.arange(11,16,dtype=int)
        n_conds=np.append(n_conds,[0]*self.settings['attn_task']['n_blanks_per_block']) # add blanks
        n_conds=np.append(n_conds,self.settings['trial_types']['low_contrast_ids']) # add an extra repetition for every low-contrast run

        # for 1up1down staircase keep staircases consecutive instead of suffling
        self.conds=np.asarray(sorted(np.concatenate([rng.permutation(n_conds) for i in range(self.settings['attn_task']['n_blocks'])])))# randomize and repeat

        # hackey but this is the flow I want for the 1 up 1 down... shuffle within stair types, insert blank bw stairtypes
        print(self.conds, '\n', type(self.conds))
        print(self.conds[np.where((self.conds > 0) & (self.conds < 14))[0]])
        self.conds=np.concatenate((rng.permutation(self.conds[np.where((self.conds > 0) & (self.conds < 14))[0]]), self.conds[np.where(self.conds >= 14)[0]]))
        self.conds=np.insert(self.conds,np.where(self.conds == 14)[0][0],0) # insert blank at first 14
        self.conds=np.insert(self.conds,np.where(self.conds == 15)[0][0],0) # insert blank at first 15
        
        self.conds=np.concatenate((np.zeros(self.settings['attn_task']['start_blanks']),self.conds)).astype(int) # add start_blanks
        self.conds=np.concatenate((self.conds,np.zeros(self.settings['attn_task']['end_blanks']))).astype(int) # add end
        np.save(opj(self.output_dir, self.output_str+'_trials.npy'),self.conds)
        self.n_trials = len(self.conds)
        self.stim_per_trial = self.settings['attn_task']['stim_per_trial']
        self.n_stim = self.n_trials * self.stim_per_trial
        self.trials = []
        self.scanner_sync = self.settings['PRF stimulus settings']['Scanner sync']
        self.resp_blue=self.settings['attn_task']['resp_keys'][0]
        self.resp_pink=self.settings['attn_task']['resp_keys'][1]
        self.behTypes=self.settings['trial_types']['response_types']
        self.behTypeLoc={}

        print(f'Number of trials: {self.n_trials}')
        print(f"Number of volumes from settings: {self.settings['mri']['volumes']}")

        if self.settings['operating system'] == 'mac':  # to compensate for macbook retina display
            self.screen = np.array([self.win.size[0], self.win.size[1]]) / 2
        else:
            self.screen = np.array([self.win.size[0], self.win.size[1]])

        #if we are scanning, here I set the mri_trigger manually to the 't'. together with the change in trial.py, this ensures syncing
        if self.scanner_sync == True:
            self.mri_trigger='t'

        if self.settings['PRF stimulus settings']['Screenshot']==True:
            self.screen_dir=output_dir+'/'+output_str+'_Screenshots'
            if not os.path.exists(self.screen_dir):
                os.mkdir(self.screen_dir)

    def create_stimuli(self):
                
        self.hemistim = HemiFieldStim(session=self, 
                                        angular_cycles=self.settings['radial'].get('angular_cycles'), 
                                        angular_res=self.settings['radial'].get('angular_res'), 
                                        radial_cycles=self.settings['radial'].get('radial_cycles'), 
                                        border_radius=self.settings['radial'].get('border_radius'), 
                                        pacman_angle=self.settings['radial'].get('pacman_angle'), 
                                        n_mask_pixels=self.settings['radial'].get('n_mask_pixels'), 
                                        frequency=self.settings['radial'].get('frequency'),
                                        outer_radius=self.settings['radial'].get('outer_radius'),
                                        inner_radius=self.settings['radial'].get('inner_radius'))

        self.largeAF = AttSizeStim(self,
                                   n_sections=self.settings['large_task']['n_sections'],
                                   ecc_min=self.settings['small_task']['radius']*5,
                                   ecc_max= np.sqrt((tools.monitorunittools.pix2deg(self.screen[0],self.monitor)/2)**2 \
                                            + (tools.monitorunittools.pix2deg(self.screen[0],self.monitor)/2)**2), # radius
                                   n_rings=self.settings['large_task']['n_rings'],
                                   row_spacing_factor=self.settings['large_task']['row_spacing'],
                                   opacity=self.settings['large_task']['opacity'],
                                   color1=self.settings['large_task']['color1'],
                                   color2=self.settings['large_task']['color2'],
                                   jitter=self.settings['large_task']['jitter'])

        self.smallAF = AttSizeStim(self,
                                   n_sections=self.settings['small_task']['n_sections'],
                                   ecc_min=0,
                                   ecc_max=self.settings['small_task']['radius'],
                                   n_rings=self.settings['small_task']['n_rings'],
                                   row_spacing_factor=self.settings['small_task']['row_spacing'],
                                   opacity=self.settings['small_task']['opacity'],
                                   color1=self.settings['small_task']['color1'],
                                   color2=self.settings['small_task']['color2'])

        self.fix_circle = FixationStim(self)
        self.cross_fix = cross_fixation(self.win, 0.2, (-1, -1, -1), opacity=0.5)

        # create fixation lines
        self.line1 = visual.Line(win=self.win,
                                 units="pix",
                                 lineColor=self.settings['fixation stim']['line_color'],
                                 lineWidth=self.settings['fixation stim']['line_width'],
                                 contrast=self.settings['fixation stim']['contrast'],
                                 start=[-self.screen[1], self.screen[1]],
                                 end=[self.screen[1], -self.screen[1]]
                                 )

        self.line2 = visual.Line(win=self.win,
                                 units="pix",
                                 lineColor=self.settings['fixation stim']['line_color'],
                                 lineWidth=self.settings['fixation stim']['line_width'],
                                 contrast=self.settings['fixation stim']['contrast'],
                                 start=[-self.screen[1], -self.screen[1]],
                                 end=[self.screen[1], self.screen[1]]
                                 )
        
        # create cue lines

        self.cueStim= cueStim(self,
                               radius=self.settings['small_task']['radius'],
                               lineColor=self.settings['cue']['color'],
                               lineWidth=self.settings['cue']['line_width'],
                               lineLength=self.settings['cue']['lineLength']
                               )
        
        self.cue_line1 = visual.Line(win=self.win,
                                 units="deg",
                                 lineColor=1,
                                 lineWidth=self.settings['fixation stim']['line_width'],
                                 contrast=self.settings['fixation stim']['contrast'],
                                 pos=(0,self.settings['small_task']['radius']),
                                 start=[self.settings['small_task']['radius'],-self.settings['cue']['lineLength']/2],
                                 end=[self.settings['small_task']['radius'],self.settings['cue']['lineLength']/2]
                                 )
        self.cue_line2 = visual.Line(win=self.win,
                                 units="deg",
                                 lineColor=1,
                                 lineWidth=self.settings['fixation stim']['line_width'],
                                 contrast=self.settings['fixation stim']['contrast'],
                                 pos=(self.settings['small_task']['radius'],0),
                                 start=[-self.settings['small_task']['radius'],-self.settings['cue']['lineLength']/2],
                                 end=[-self.settings['small_task']['radius'],self.settings['cue']['lineLength']/2]
                                 )
        

    def create_staircase(self):
        """set up multi condition staircase
        """
        stair_info= {}
        self.stairs={}
        self.behTypeLoc={behType:[] for behType in self.behTypes}
        
        for k,v in self.settings['staircase']['info'].items():
            stair_info[k]=v

        for behType in self.responses.keys():
            if behType == 'blank':
                continue
            self.stairs[behType]=data.StairHandler(startVal=stair_info['startVal'],
                                                    minVal=stair_info['minVal'],
                                                    maxVal=stair_info['maxVal'],
                                                    stepSizes=stair_info['stepSizes'],
                                                    stepType=stair_info['stepType'],
                                                    nUp=stair_info['nUp'],
                                                    nDown=stair_info['nDown'],
                                                    nTrials=stair_info['nTrials'],
                                                    name=behType)
            # self.stairs[behType].discreet=[]
        
            # make it easier to update next trial of same response types
            self.behTypeLoc[behType]=np.where(np.array(self.behTypeTrials)==self.behTypes.index(behType))[0]

    def create_trials(self):
        """Create array containing a trial object for every trial,
        adds important information to each trial in the form of parameters,
        changes trial duration for sync triggers
        Some important variables for the running of the experiment also declared here:
            - self.responses (dict): dictionary to store responses, each key is behType
            - self.behTypeTrials (arr): array of length nTrials, each element is the behType_id (from settings) of that trial
        """

        self.responses={}
        self.behTypeTrials = []
        self.responses['blank'] = -1 * np.ones(self.n_trials)

        for behType in self.behTypes:
            self.responses[behType] = -1 * np.ones(self.n_trials)

        for i in range(self.n_trials):
            phase_durations = copy.copy(self.settings['attn_task']['phase_durations']) # don't overwrite settings dict
            
            # sync start blanks to MRI 
            if (i < self.settings['attn_task']['start_blanks']) & (self.scanner_sync):
                phase_durations=[100]
                sync_trigger=True
            # sync end blanks to MRI 
            elif (i >= self.n_trials - self.settings['attn_task']['end_blanks']) & (self.scanner_sync):
                phase_durations=[100]
                sync_trigger=True
            # sync to MRI trigger first attn trial and every nth trial
            elif ((i - self.settings['attn_task']['start_blanks'] +1) % self.settings['attn_task']['sync_trial'] == 0) & (i != 0) & (self.scanner_sync):
                phase_durations[-1]=100 
                sync_trigger=True
            else:
                sync_trigger=False

            # treat blank trials a bit differently
            if self.conds[i]==0:
                self.behTypeTrials.append(-1)

                parameters=dict()
                parameters['task']='blank'
                parameters['large_opacity']= 0
                parameters['mapper_contrast']= 0
                parameters['large_balance'] = 0
                parameters['small_balance'] = 0
                parameters['response_type'] = 'blank'
            
                self.trials.append(BlankTrial(session=self,
                                            trial_nr=i,
                                            phase_durations=phase_durations,
                                            sync_trigger=sync_trigger,
                                            parameters=parameters
                                                ))
                continue
            else:
                self.behTypeTrials.append(self.settings['trial_types'][str(self.conds[i])]['response_type_id'])
                
                # paramaters to describe each trial
                parameters=dict()
                parameters['task']=self.settings['trial_types'][str(self.conds[i])]['cue']
                large_opacity=self.settings['trial_types'][str(self.conds[i])]['large_opacity']
                mapper_contrast=self.settings['trial_types'][str(self.conds[i])]['mapper_contrast']
                parameters['large_opacity']=self.settings['trial_types'][f'{large_opacity}_task_opacity']
                parameters['mapper_contrast']=self.settings['trial_types'][f'{mapper_contrast}_mapper_contrast']
                # parameters['large_balance'] = self.large_balances[i]
                # parameters['small_balance'] = self.small_balances[i]
                parameters['response_type'] = self.behTypes[self.settings['trial_types'][str(self.conds[i])]['response_type_id']]

                self.trials.append(AttnTrial(session=self,
                                            trial_nr=i,
                                            phase_durations=phase_durations,
                                            sync_trigger=sync_trigger,
                                            parameters=parameters
                                            ))
                
    
    def draw_small_stimulus(self,balance=None,opacity=1):
        self.stim_nr = self.current_trial.trial_nr
        self.fix_circle.draw(0, radius=self.settings['small_task'].get('radius'))
        self.smallAF.draw(balance, self.stim_nr,opacity)
    
    def draw_large_stimulus(self,balance=None,opacity=1):
        self.stim_nr = self.current_trial.trial_nr
        self.largeAF.draw(balance, self.stim_nr,opacity)

    def draw_mapper(self,contrast=1):
        self.hemistim.draw(contrast)
        # self.inner_mask_stim.draw()

        # need to redraw fixations because the masking covers it
        self.line1.draw() # fixation guides
        self.line2.draw()

    def run(self):
        """run the session"""
        # cycle through trials

        if self.eyetracker_on:
            self.calibrate_eyetracker()

        self.line1.draw()
        self.line2.draw()
        self.fix_circle.draw(0, radius=self.settings['small_task'].get('radius'))
        self.display_text('', keys=self.settings['mri'].get('sync', 't'))

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        
        for trial_idx in range(len(self.trials)):
            self.current_trial = self.trials[trial_idx]
            behType=self.current_trial.parameters['response_type'] #same as self.behTypeTrials[trial_idx]
            self.current_trial.behType=behType

            # skip blanks for staircase
            try:
                intensity=self.stairs[behType].intensity
                self.current_trial.parameters['large_balance']=intensity
                self.current_trial.parameters['small_balance']=intensity
            except KeyError:
                pass

            self.current_trial_start_time = self.clock.getTime()
            print(f'Trial {trial_idx}, time {self.current_trial_start_time}')
            # print(f'Small balance: {self.current_trial.parameters["small_balance"]}, \nLarge balance: {self.current_trial.parameters["large_balance"]}')
            
            self.current_trial.run()    
            # self.win.getMovieFrame()
            
            if self.conds[trial_idx] != 0:
                resp=self.responses[behType][self.current_trial.trial_nr]
                # resp=0 if resp == -1 else resp # change no response to incorrect 
                print('\n', behType)
                print('This intensity: ', self.stairs[behType].intensity)
                print('resp ', resp)

                if resp != -1: #skip staircase for no response
                    self.stairs[behType].intensities.append(self.stairs[behType].intensity)
                    self.stairs[behType].addResponse(resp,intensity=self.stairs[behType].intensity)

                print('Next intensity: ', self.stairs[behType].intensity)
                print('Intensities: ', self.stairs[behType].intensities,'\n')
        
        np.save(opj(self.output_dir, self.output_str+'_trials.npy'),self.conds)
        with open(opj(self.output_dir, self.output_str+'_responses.npy'), 'wb') as f:
            pickle.dump(self.responses, f)
        
        for beh in self.behTypes:
            self.stairs[beh].saveAsText(opj(self.output_dir, self.output_str+f'_{beh}.txt'))
            with open(opj(self.output_dir, self.output_str+f'_{beh}.npy'), 'wb') as f:
                pickle.dump(self.stairs[beh], f)

        # if self.settings['PRF stimulus settings']['Screenshot']==True:
        #     self.win.saveMovieFrames(opj(self.screen_dir, self.output_str+'_Screenshot.png'))
            
        self.close()
        self.quit()
