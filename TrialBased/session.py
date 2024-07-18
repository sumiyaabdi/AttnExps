#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:05:10 2019

@author: marcoaqil
"""

import numpy as np
import os

import sys
sys.path.append('../../exptools2')

from psychopy import visual, tools
from psychopy.visual import filters

from exptools2.core import Session, PylinkEyetrackerSession
from trial import BaselineTrial, PRFTrial, PsychophysTrial, AttnTrial, BlankTrial
from stim import PRFStim, AttSizeStim, FixationStim,cross_fixation,HemiFieldStim
from utils import create_stim_list, get_stim_nr,psyc_stim_list


opj = os.path.join

class AttnSession(PylinkEyetrackerSession):
    def __init__(self, output_str, 
                 output_dir, 
                 settings_file, 
                 eyetracker_on=True):

        super().__init__(output_str=output_str, 
                         output_dir=output_dir, 
                         settings_file=settings_file, 
                         eyetracker_on=eyetracker_on)

        self.n_trials = 150
        self.stim_per_trial = self.settings['attn_task']['stim_per_trial']
        self.n_stim = self.n_trials * self.stim_per_trial
        self.trials = []

        self.large_balances = psyc_stim_list(self.settings['psychophysics']['large_range'], 
                                            self.n_stim, self.settings['large_task']['default_balance'])
        
        self.small_balances = psyc_stim_list(self.settings['psychophysics']['small_range'], 
                                            self.n_stim, self.settings['small_task']['default_balance'])
        
        if self.settings['operating system'] == 'mac':  # to compensate for macbook retina display
            self.screen = np.array([self.win.size[0], self.win.size[1]]) / 2
        else:
            self.screen = np.array([self.win.size[0], self.win.size[1]])


        #if we are scanning, here I set the mri_trigger manually to the 't'. together with the change in trial.py, this ensures syncing
        if self.settings['PRF stimulus settings']['Scanner sync']==True:
            self.mri_trigger='t'

        if self.settings['PRF stimulus settings']['Screenshot']==True:
            self.screen_dir=output_dir+'/'+output_str+'_Screenshots'
            if not os.path.exists(self.screen_dir):
                os.mkdir(self.screen_dir)

    def create_stimuli(self):
        outer_mask = filters.makeMask(matrixSize=self.screen[0],
                                shape='raisedCosine', 
                                radius=np.array([self.screen[1]/self.screen[0], 1.0]),
                                center=(0.0, 0.0), 
                                range=[-1, 1], 
                                fringeWidth=0.02
                                )

        self.outer_mask_stim = visual.GratingStim(self.win, 
                                        mask=-outer_mask, 
                                        tex=None, 
                                        units='pix',
                                        size=self.screen,
                                        pos = np.array((0.0,0.0)), 
                                        color = [0,0,0])
        
        self.inner_mask_stim = visual.Circle(self.win, 
                                    contrast=1,
                                    radius=2, 
                                    fillColor=0, 
                                    edges='circle')
        
        self.hemistim = HemiFieldStim(session=self, 
                                        angular_cycles=self.settings['radial'].get('angular_cycles'), 
                                        angular_res=self.settings['radial'].get('angular_res'), 
                                        radial_cycles=self.settings['radial'].get('radial_cycles'), 
                                        border_radius=self.settings['radial'].get('border_radius'), 
                                        pacman_angle=self.settings['radial'].get('pacman_angle'), 
                                        n_mask_pixels=self.settings['radial'].get('n_mask_pixels'), 
                                        frequency=self.settings['radial'].get('frequency'))

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
    def create_trials(self):
        """include each trial detail (i.e. trial type, task, cue, stimulus displyed)
        """

        for i in range(self.n_trials):
            if i in self.settings['attn_task']['blank_trials_pos']:
                self.trials.append(BlankTrial(session=self,
                                            trial_nr=i,
                                                ))
                continue
            else:
                self.trials.append(AttnTrial(session=self,
                                            trial_nr=i,
                                            cue='S',
                                            draw_large=True,
                                            large_opacity=1,
                                            draw_mapper=True,
                                            mapper_contrast=0.8
                                            ))
    
    def draw_small_stimulus(self):
        self.stim_nr = self.current_trial.trial_nr
        self.fix_circle.draw(0, radius=self.settings['small_task'].get('radius'))
        self.smallAF.draw(self.small_balances[self.stim_nr], self.stim_nr)
    
    def draw_large_stimulus(self,opacity=1):
        self.stim_nr = self.current_trial.trial_nr
        # self.fix_circle.draw(0, radius=self.settings['small_task'].get('radius'))
        self.largeAF.draw(self.large_balances[self.stim_nr], self.stim_nr)
        # self.smallAF.draw(self.small_balances[self.stim_nr], self.stim_nr)

    def draw_mapper(self,contrast=1):
        self.hemistim.draw()
        self.inner_mask_stim.draw()
        self.outer_mask_stim.draw()

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
            self.current_trial_start_time = self.clock.getTime()
            print(f'Trial {trial_idx}, time {self.current_trial_start_time}')
            self.current_trial.run()
        
        print('Total subject responses: %d'%self.total_responses)
        np.save(opj(self.output_dir, self.output_str+'_simple_response_data.npy'), {'Total subject responses':self.total_responses})
        
        
        if self.settings['PRF stimulus settings']['Screenshot']==True:
            self.win.saveMovieFrames(opj(self.screen_dir, self.output_str+'_Screenshot.png'))
            
        self.close()