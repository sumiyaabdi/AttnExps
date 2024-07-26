#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:06:36 2019

@author: marcoaqil
"""

from exptools2.core.trial import Trial
from psychopy import event
from utils import get_stim_nr
import numpy as np
import pickle
import os

opj = os.path.join

class AttnTrial(Trial):
    def __init__(self, session, 
                trial_nr, 
                phase_durations,
                sync_trigger,
                cue='F',
                draw_large=True,
                large_opacity='high',
                draw_mapper=True,
                mapper_contrast='high',
                parameters={},
                *args, 
                **kwargs):

        super().__init__(session, trial_nr, phase_durations=phase_durations, *args, **kwargs)

        self.cue=parameters['task']
        self.draw_large=draw_large
        self.draw_mapper=draw_mapper
        self.parameters = parameters
        self.sync_trigger = sync_trigger

    def draw(self, *args, **kwargs):
        """Draws stimuli"""

        self.session.line1.draw() # fixation guides
        self.session.line2.draw()
        self.session.fix_circle.draw(0, radius=self.session.settings['small_task'].get('radius'))

        if self.phase == 0:
            if self.cue.upper() == 'S':
                print('s cue')
                self.session.cueStim.draw_cardinal()
            elif self.cue.upper() == 'L':
                print('s cue')
                self.session.cueStim.draw_diagonal()

        elif (self.phase % 2 == 0):
            self.session.win.getMovieFrame()
            if self.draw_mapper:
                self.session.draw_mapper(contrast=self.parameters['mapper_contrast'])
            if self.draw_large:
                self.session.draw_large_stimulus(opacity=self.parameters['large_opacity'])
            
            self.session.draw_small_stimulus()

    def get_events(self):
        """ ATTNTRIAL method. Logs responses/triggers """
        events = event.getKeys(timeStamped=self.session.clock)
        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?

                #saves in the event of a quit
                np.save(opj(self.session.output_dir, self.session.output_str+'_trials.npy'),self.session.conds)
                with open(opj(self.session.output_dir, self.session.output_str+'_responses.npy'), 'wb') as f:
                    pickle.dump(self.session.responses, f)

                self.session.close()
                self.session.quit()

            for key, t in events:
                if key == self.session.mri_trigger:
                    event_type = 'pulse'
                    if self.sync_trigger:
                        self.exit_phase=True

                else:
                    event_type = 'response'

                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = key

                # for param, val in self.parameters.items():
                    # self.session.global_log.loc[idx, param] = val
                for param, val in self.parameters.items():  # add parameters to log
                    if type(val) == np.ndarray or type(val) == list:
                        for i, x in enumerate(val):
                            self.session.global_log.loc[idx, param+'_%4i'%i] = x 
                    else:       
                        self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{key}_time-{t}'
                    self.session.tracker.sendMessage(msg)

                if key != self.session.mri_trigger:
                    self.last_resp = key
                    self.last_resp_onset = t

                    # append response (correct / incorrect) to specific trial type on if resp appears after stimulus does
                    if self.session.clock.getTime() > (self.start_trial + self.phase_durations[0]+self.phase_durations[1]):
                        self.session.responses[self.parameters['response_type']][self.trial_nr] = self.check_correct(key)
                    # self.session.responses[self.parameters['response_type']][self.trial_nr] = self.check_correct(key)
        return events
    
    def check_correct(self, response):
        
        task= 'large' if self.parameters['task'] == 'l' else 'small'
        balance = self.parameters[f'{task}_balance']

        if balance < 0.5:
            correct_response=self.session.resp_pink
        elif balance > 0.5:
            correct_response=self.session.resp_blue
        else:
            correct_response=None

        if response == correct_response:
            return 1
        else:  
            return 0


class BlankTrial(AttnTrial):
    def __init__(self, 
                 session, 
                 trial_nr, 
                 phase_durations,
                 sync_trigger,
                 *args, **kwargs):

        super().__init__(session, trial_nr, phase_durations, sync_trigger, *args, **kwargs)

    def draw(self,*args, **kwargs):
        self.session.line1.draw() # fixation guides
        self.session.line2.draw()
        self.session.fix_circle.draw(0, radius=self.session.settings['small_task'].get('radius'))

# class PsychophysTrial(Trial):
#     def __init__(self, session, trial_nr, bar_orientation, bar_position_in_ori,
#                  bar_direction, *args, **kwargs):

#         self.session = session
#         self.bar_orientation = bar_orientation
#         self.bar_position_in_ori = bar_position_in_ori
#         self.bar_direction = bar_direction
#         phase_durations = [self.session.settings['mri']['TR']/4,100,0.3]

#         super().__init__(session, trial_nr, phase_durations, *args, **kwargs)

#     def draw(self, *args, **kwargs):
#         # draw attention task stimulus & mapper

#         """ Draws stimuli """

#         self.session.hemistim.draw()    
#         self.session.inner_mask_stim.draw()
#         self.session.outer_mask_stim.draw()

#         self.session.line1.draw() # fixation guides
#         self.session.line2.draw()


#         if self.phase == 0:
#             self.session.draw_attn_stimulus()
#         elif self.phase == 1:
#             self.session.fix_circle.draw(0, radius=self.session.settings['small_task'].get('radius'))
#         elif self.phase == 2:
#             self.session.fix_circle.draw(0, radius=self.session.settings['small_task'].get('radius'))

#     def get_events(self):
#         """ Logs responses/triggers """
#         events = event.getKeys(timeStamped=self.session.clock)
#         # waitKeys = event.waitKeys(keyList=['left','right'], timeStamped=self.session.clock)
#         if events:
#             if 'q' in [ev[0] for ev in events]:  # specific key in settings?

#                 np.save(opj(self.session.output_dir, self.session.output_str + '_simple_response_data.npy'),
#                         {'Total subject responses': self.session.total_responses})

#                 if self.session.settings['PRF stimulus settings']['Screenshot'] == True:
#                     self.session.win.saveMovieFrames(
#                         opj(self.session.screen_dir, self.session.output_str + '_Screenshot.png'))

#                 self.session.close()
#                 self.session.quit()

#             for key, t in events:
#                 event_type = 'response'
#                 self.session.total_responses += 1
#                 self.exit_phase = True

#                 # if key == self.session.mri_trigger:
#                 #     event_type = 'pulse'
#                 #     # marco edit. the second bit is a hack to avoid double-counting of the first t when simulating a scanner
#                 #     if self.session.settings['PRF stimulus settings']['Scanner sync'] == True and t > 0.1:
#                 #         # ideally, for speed, would want  getMovieFrame to be called right after the first winflip.
#                 #         # but this would have to be dun from inside trial.run()
#                 #         if self.session.settings['PRF stimulus settings']['Screenshot'] == True:
#                 #             self.session.win.getMovieFrame()

#                 idx = self.session.global_log.shape[0]
#                 self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
#                 self.session.global_log.loc[idx, 'onset'] = t
#                 self.session.global_log.loc[idx, 'event_type'] = event_type
#                 self.session.global_log.loc[idx, 'phase'] = self.phase
#                 self.session.global_log.loc[idx, 'response'] = key
#                 self.session.global_log.loc[idx, 'large_prop'] = self.session.large_balances[get_stim_nr(self.trial_nr, self.phase, self.session.stim_per_trial)]
#                 self.session.global_log.loc[idx, 'small_prop'] = self.session.small_balances[get_stim_nr(self.trial_nr, self.phase, self.session.stim_per_trial)]

#                 for param, val in self.parameters.items():
#                     self.session.global_log.loc[idx, param] = val

#                 if key != self.session.mri_trigger:
#                     self.last_resp = key
#                     self.last_resp_onset = t
