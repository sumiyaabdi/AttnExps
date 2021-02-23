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

from exptools2.core.session import Session
from trial import PRFTrial
from stim import PRFStim, AttSizeStim, FixationStim, cross_fixation
from utils import create_stim_list


opj = os.path.join



class PRFSession(Session):

    def __init__(self, output_str, output_dir, settings_file):

        """

        Parameters
        ----------
        output_str
        output_dir
        settings_file
        """
        
        super().__init__(output_str=output_str, output_dir=output_dir, settings_file=settings_file)

        self.color_range = self.settings['attn stim']['color_range']
        self.fix_range = self.settings['fixation stim']['gray_range']
        print(f'self.fix_range: {self.fix_range}')
        self.bar_orientations = np.array(self.settings['PRF stimulus settings']['Bar orientations'])
        self.n_trials = 5 + self.settings['PRF stimulus settings']['Bar pass steps'] \
                        * len(np.where(self.bar_orientations != -1)[0]) \
                        + self.settings['PRF stimulus settings']['Blanks length'] \
                        * len(np.where(self.bar_orientations == -1)[0])
        self.stim_per_trial = self.settings['attn stim']['stim_per_trial']
        self.n_stim = self.n_trials * self.stim_per_trial
        self.trials = []

        print(f'Monitor size (deg): {tools.monitorunittools.pix2deg(self.win.size[1],self.monitor)}')

        # # set size of display
        # if self.settings['window']['display'] == 'square':
        #     self.screen = np.array([self.win.size[1], self.win.size[1]])

        # elif self.settings['window']['display'] == 'rectangle':
        #     self.screen = np.array([self.win.size[0], self.win.size[1]])

        if self.settings['operating system'] == 'mac':  # to compensate for macbook retina display
            self.screen = np.array([self.win.size[0], self.win.size[1]]) / 2
        else:
            self.screen = np.array([self.win.size[0], self.win.size[1]])


        #if we are scanning, here I set the mri_trigger manually to the 't'. together with the change in trial.py, this ensures syncing
        if self.settings['mri']['topup_scan']==True:
            self.topup_scan_duration=self.settings['mri']['topup_duration']
        
        if self.settings['PRF stimulus settings']['Scanner sync']==True:
            self.bar_step_length = self.settings['mri']['TR']
            self.mri_trigger='t'

                     
        else:
            self.bar_step_length = self.settings['PRF stimulus settings']['Bar step length']
            
        if self.settings['PRF stimulus settings']['Screenshot']==True:
            self.screen_dir=output_dir+'/'+output_str+'_Screenshots'
            if not os.path.exists(self.screen_dir):
                os.mkdir(self.screen_dir)


    def create_stimuli(self):

        #generate PRF stimulus
        self.prf_stim = PRFStim(session=self, 
                        squares_in_bar=self.settings['PRF stimulus settings']['Squares in bar'], 
                        bar_width_deg=self.settings['PRF stimulus settings']['Bar width in degrees'],
                        flicker_frequency=self.settings['PRF stimulus settings']['Checkers motion speed'])#self.deg2pix(self.settings['prf_max_eccentricity']))    

        if self.settings['operating system'] == 'mac':
            mask_size = [self.win.size[0]/2,self.win.size[1]/2]
        else:
            mask_size = [self.win.size[0],self.win.size[1]]

        #generate raised cosine alpha mask
        mask = filters.makeMask(matrixSize=self.screen[0],
                                shape='raisedCosine', 
                                radius=np.array([self.screen[1]/self.screen[0], 1.0]),
                                center=(0.0, 0.0), 
                                range=[-1, 1], 
                                fringeWidth=0.02
                                )

        self.mask_stim = visual.GratingStim(self.win, 
                                        mask=-mask, 
                                        tex=None, 
                                        units='pix',
                                        size=mask_size,
                                        pos = np.array((0.0,0.0)), 
                                        color = [0,0,0])

        

        self.largeAF = AttSizeStim(self,
                                   n_sections=self.settings['attn stim']['number of sections'],
                                   ecc_min=self.settings['fixation stim']['radius'] + 0.2,
                                   ecc_max= tools.monitorunittools.pix2deg(self.win.size[1],self.monitor)/2 + 0.5, # radius
                                   n_rings=self.settings['attn stim']['number of rings'],
                                   row_spacing_factor=self.settings['attn stim']['row spacing factor'],
                                   opacity=self.settings['attn stim']['opacity'])

        self.smallAF = FixationStim(self)
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
        """creates trials by setting up prf stimulus sequence
        create bar orientation list at each TR (this can be done in many different ways according to necessity)
        for example, currently blank periods have same length as bar passes. this can easily be changed here"""

        self.correct_responses = 0
        self.total_responses = 0

        print("Expected number of TRs: %d"%self.n_trials)

        steps_array=self.settings['PRF stimulus settings']['Bar pass steps']*np.ones(len(self.bar_orientations))
        blanks_array=self.settings['PRF stimulus settings']['Blanks length']*np.ones(len(self.bar_orientations))
        repeat_times=np.where(self.bar_orientations == -1, blanks_array, steps_array).astype(int)
        self.bar_orientation_at_TR = np.concatenate((-1*np.ones(5), np.repeat(self.bar_orientations, repeat_times)))
        bar_pos_array = self.screen[1]*np.linspace(-0.5,0.5, self.settings['PRF stimulus settings']['Bar pass steps'])
        blank_array = np.zeros(self.settings['PRF stimulus settings']['Blanks length'])
        
        #the 5 empty trials at beginning
        self.bar_pos_in_ori=np.zeros(5)
        
        #bar position at TR
        for i in range(len(self.bar_orientations)):
            if self.bar_orientations[i]==-1:
                self.bar_pos_in_ori=np.append(self.bar_pos_in_ori, blank_array)
            else:
                self.bar_pos_in_ori=np.append(self.bar_pos_in_ori, bar_pos_array)
     
        #random bar direction at each step. could also make this time-based
        self.bar_direction_at_TR = np.round(np.random.rand(self.n_trials))


        ### attn trial details ###
        signal = self.n_stim / self.settings['attn stim']['signal']

        self.color_balances = create_stim_list(self.n_stim, signal, self.color_range)
        self.fix_colors = create_stim_list(self.n_stim, signal, self.fix_range)

        print(f'color_balances: {self.color_balances} \nLEN color balances:{len(self.color_balances)}')
        print(f'n.stim: {self.n_stim}')


        # trial list
        for i in range(self.n_trials):
            self.trials.append(PRFTrial(session=self,
                                        trial_nr=i,

                                        bar_orientation=self.bar_orientation_at_TR[i],
                                        bar_position_in_ori=self.bar_pos_in_ori[i],
                                        bar_direction=self.bar_direction_at_TR[i]
                                        ))

        # times for dot color change. continue the task into the topup
        self.total_time = self.n_trials * self.bar_step_length

        if self.settings['mri']['topup_scan'] == True:
            self.total_time += self.topup_scan_duration

    def draw_prf_stimulus(self):
        #this timing is only used for the motion of checkerboards inside the bar. it does not have any effect on the actual bar motion
        present_time = self.clock.getTime()
        
        #present_trial_time = self.clock.getTime() - self.current_trial_start_time
        prf_time = present_time #/ (self.bar_step_length)

        # draw the bar at the required orientation for this TR, unless the orientation is -1, code for a blank period
        if self.current_trial.bar_orientation != -1:
            self.prf_stim.draw(time=prf_time,
                               pos_in_ori=self.current_trial.bar_position_in_ori,
                               orientation=self.current_trial.bar_orientation,
                               bar_direction=self.current_trial.bar_direction)

    def draw_attn_stimulus(self, phase):
        self.stim_nr = self.current_trial.trial_nr * self.stim_per_trial + int((phase-1)/2)
        self.largeAF.draw(self.color_balances[self.stim_nr], self.stim_nr)
        self.smallAF.draw(self.fix_colors[self.stim_nr],
                          radius=self.settings['fixation stim'].get('radius'))
  


    def run(self):
        """run the session"""
        # cycle through trials
        self.display_text('Waiting for scanner', keys=self.settings['mri'].get('sync', 't'))

        self.start_experiment()
        
        for trial_idx in range(len(self.trials)):
            self.current_trial = self.trials[trial_idx]
            self.current_trial_start_time = self.clock.getTime()
            self.current_trial.run()
        
<<<<<<< HEAD
=======
        print('Total subject responses: %d'%self.total_responses)
        np.save(opj(self.output_dir, self.output_str+'_simple_response_data.npy'), {'Total subject responses':self.total_responses})
        
        
>>>>>>> d9ab0ea3fe91ae2231d680222bef95968ab6e748
        if self.settings['PRF stimulus settings']['Screenshot']==True:
            self.win.saveMovieFrames(opj(self.screen_dir, self.output_str+'_Screenshot.png'))
            
        self.close()

        

