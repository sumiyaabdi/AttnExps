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
import os

opj = os.path.join

class BaselineTrial(Trial):
    def __init__(self, session, trial_nr):
        self.session = session
        self.trial_nr = trial_nr
        
        if self.session.settings['PRF stimulus settings']['Scanner sync']==True:
            if self.trial_nr == self.session.n_trials-1:
                print(self.trial_nr, self.session.n_trials-1)
                self.phase_durations = [1.5]
            else:
                self.phase_durations = [50] #dummy value
        else:
            self.phase_durations = [self.session.settings['mri']['TR']]
        
        super().__init__(session, trial_nr,self.phase_durations,verbose=False)
    
    def draw(self):
        self.session.line1.draw()
        self.session.line2.draw()
        self.session.fix_circle.draw(0,radius=self.session.settings['fixation_stim']['radius'])
        
    def get_events(self):
         """ Logs responses/triggers """
         events = event.getKeys(timeStamped=self.session.clock)
         if events:
             if 'q' in [ev[0] for ev in events]:  # specific key in settings?
                 np.save(opj(self.session.output_dir, self.session.output_str+'_simple_response_data.npy'),
                         {'Total subject responses': self.session.total_responses})
                 
                 if self.session.settings['PRF stimulus settings']['Screenshot']==True:
                     self.session.win.saveMovieFrames(opj(self.session.screen_dir, self.session.output_str+'_Screenshot.png'))
                     
                 self.session.close()
                 self.session.quit()
 
             for key, t in events:
                if t >= 390:
                    self.exit_phase = True

                if key == self.session.mri_trigger:
                    event_type = 'pulse'
                        #marco edit. the second bit is a hack to avoid double-counting of the first t when simulating a scanner
                    if self.session.settings['PRF stimulus settings']['Scanner sync']==True and t>0.1:
                        self.exit_phase=True
                            #ideally, for speed, would want  getMovieFrame to be called right after the first winflip.
                            #but this would have to be dun from inside trial.run()
                        if self.session.settings['PRF stimulus settings']['Screenshot']==True:
                                self.session.win.getMovieFrame()
                else:
                    event_type = 'response'
                    self.session.total_responses += 1

                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = key
                # self.session.global_log.loc[idx, 'letter'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].text
                # self.session.global_log.loc[idx, 'color'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color
                # self.session.global_log.loc[idx, 'italic'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].italic

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val

                if key != self.session.mri_trigger:
                    self.last_resp = key
                    self.last_resp_onset = t

class PsychophysTrial(Trial):
    def __init__(self, session, trial_nr, bar_orientation, bar_position_in_ori,
                 bar_direction, *args, **kwargs):

        self.session = session
        self.bar_orientation = bar_orientation
        self.bar_position_in_ori = bar_position_in_ori
        self.bar_direction = bar_direction

        phase_durations = self.session.settings['psychophysics']['phase_dur']

        super().__init__(session, trial_nr, phase_durations, *args, **kwargs)

    def draw(self, *args, **kwargs):
        # draw bar stimulus and circular (raised cosine) aperture from Session class
        """ Draws stimuli """

        
        self.session.draw_prf_stimulus()
        self.session.mask_stim.draw()

        self.session.line1.draw()
        self.session.line2.draw()

        self.session.fix_circle.draw(0,radius=self.session.settings['fixation_stim']['radius'])

        if self.phase % 2 == 0:
            self.session.draw_attn_stimulus(self.trial_nr,self.phase)

    def log_phase_info(self, phase=None):
            """ Method passed to win.callonFlip, such that the
            onsets get logged *exactly* on the screen flip.
            Phase can be passed as an argument to log the onsets
            of phases that finish before a window flip (e.g.,
            phases with duration = 0, and are skipped on some
            trials).
            """
            onset = self.session.clock.getTime()

            if phase is None:
                phase = self.phase

            if phase == 0:
                self.start_trial = onset

            msg = f"\tPhase {phase} start: {onset:.5f}"

            if self.verbose:
                print(msg)

            if self.eyetracker_on:  # send msg to eyetracker
                msg = f'start_type-stim_trial-{self.trial_nr}_phase-{phase}'
                self.session.tracker.sendMessage(msg)
                # Should be log more to the eyetracker? Like 'parameters'?

            # add to global log
            idx = self.session.global_log.shape[0]
            self.session.global_log.loc[idx, 'onset'] = onset
            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
            self.session.global_log.loc[idx, 'event_type'] = self.phase_names[phase]
            self.session.global_log.loc[idx, 'phase'] = phase
            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames
            print ('trial: ', self.trial_nr, 'phase: ', self.phase)
            self.session.global_log.loc[idx, 'letter'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].text
            self.session.global_log.loc[idx, 'color_r'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[0]
            self.session.global_log.loc[idx, 'color_g'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[1]
            self.session.global_log.loc[idx, 'color_b'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[2]
            self.session.global_log.loc[idx, 'ori'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].ori

            for param, val in self.parameters.items():  # add parameters to log
                self.session.global_log.loc[idx, param] = val

            self.session.nr_frames = 0

    def get_events(self):
        """ Logs responses/triggers """
        events = event.getKeys(timeStamped=self.session.clock)
        # waitKeys = event.waitKeys(keyList=['left','right'], timeStamped=self.session.clock)

        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?

                np.save(opj(self.session.output_dir, self.session.output_str + '_simple_response_data.npy'),
                        {'Total subject responses': self.session.total_responses})

                if self.session.settings['PRF stimulus settings']['Screenshot'] == True:
                    self.session.win.saveMovieFrames(
                        opj(self.session.screen_dir, self.session.output_str + '_Screenshot.png'))

                self.session.close()
                self.session.quit()

            for key, t in events:
                if key == self.session.mri_trigger:
                    event_type = 'pulse'
                        #marco edit. the second bit is a hack to avoid double-counting of the first t when simulating a scanner
                    if self.session.settings['PRF stimulus settings']['Scanner sync']==True:
                        self.exit_phase=True
                            #ideally, for speed, would want  getMovieFrame to be called right after the first winflip.
                            #but this would have to be dun from inside trial.run()
                        if self.session.settings['PRF stimulus settings']['Screenshot']==True:
                            if self.phase == 0:
                                self.session.win.getMovieFrame()
                else:
                    event_type = 'response'
                    self.session.total_responses += 1
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = key
                self.session.global_log.loc[idx, 'letter'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].text
                self.session.global_log.loc[idx, 'color_r'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[0]
                self.session.global_log.loc[idx, 'color_g'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[1]
                self.session.global_log.loc[idx, 'color_b'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].color[2]
                self.session.global_log.loc[idx, 'ori'] = self.session.rsvp.rsvp_stim[self.trial_nr+int(self.phase/2)].ori


                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val


class PRFTrial(Trial):

    def __init__(self, session, trial_nr, bar_orientation, bar_position_in_ori,
                 bar_direction, *args, **kwargs):

        #trial number and bar parameters   
        self.bar_orientation = bar_orientation
        self.bar_position_in_ori = bar_position_in_ori
        self.bar_direction = bar_direction
        self.session = session
        self.trial_nr = trial_nr

        if self.session.settings['psychophysics']['task'] == True:
            self.phase_durations = [100]
        else:
            #here we decide how to go from each trial (bar position) to the next.
            if self.session.settings['PRF stimulus settings']['Scanner sync']==True:
                # phase duration is determined by number of stimulus
                self.phase_durations = [self.session.settings['PRF stimulus settings']['Bar step length']\
                                       /(self.session.stim_per_trial*2)]*(self.session.stim_per_trial*2-1)
                self.phase_durations.append(100) # only last phase needs dummy value to wait for 't' 
    
                if self.trial_nr == self.session.n_trials-1: # no long last phase duration for last trial
                    self.phase_durations =[self.session.settings['PRF stimulus settings']['Bar step length']\
                                       /(self.session.stim_per_trial*2)]*(self.session.stim_per_trial*2)
            else:
                #if not synced to a real or simulated scanner, take the bar pass step as length
                self.phase_durations = [self.session.settings['PRF stimulus settings']['Bar step length']\
                                       /(self.session.stim_per_trial*2)]*(self.session.stim_per_trial*2)
        super().__init__(session, trial_nr, self.phase_durations,
                         verbose=False, *args, **kwargs)

    def draw(self, *args, **kwargs):
        # draw bar stimulus and circular (raised cosine) aperture from Session class
        """ Draws stimuli """

        
        self.session.draw_prf_stimulus()
        self.session.mask_stim.draw()

        self.session.line1.draw()
        self.session.line2.draw()

        self.session.fix_circle.draw(0,radius=self.session.settings['fixation_stim']['radius'])

        if self.phase % 2 == 0:
            self.session.draw_attn_stimulus(self.phase)
    
    def log_phase_info(self, phase=None):
            """ Method passed to win.callonFlip, such that the
            onsets get logged *exactly* on the screen flip.
            Phase can be passed as an argument to log the onsets
            of phases that finish before a window flip (e.g.,
            phases with duration = 0, and are skipped on some
            trials).
            """
            onset = self.session.clock.getTime()

            if phase is None:
                phase = self.phase

            if phase == 0:
                self.start_trial = onset

            msg = f"\tPhase {phase} start: {onset:.5f}"

            if self.verbose:
                print(msg)

            if self.eyetracker_on:  # send msg to eyetracker
                msg = f'start_type-stim_trial-{self.trial_nr}_phase-{phase}'
                self.session.tracker.sendMessage(msg)
                # Should be log more to the eyetracker? Like 'parameters'?

            # add to global log
            idx = self.session.global_log.shape[0]
            self.session.global_log.loc[idx, 'onset'] = onset
            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
            self.session.global_log.loc[idx, 'event_type'] = self.phase_names[phase]
            self.session.global_log.loc[idx, 'phase'] = phase
            self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames
            print ('trial: ', self.trial_nr, 'phase: ', self.phase)
            self.session.global_log.loc[idx, 'letter'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].text
            self.session.global_log.loc[idx, 'color_r'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[0]
            self.session.global_log.loc[idx, 'color_g'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[1]
            self.session.global_log.loc[idx, 'color_b'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[2]
            self.session.global_log.loc[idx, 'ori'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].ori

            for param, val in self.parameters.items():  # add parameters to log
                self.session.global_log.loc[idx, param] = val

            self.session.nr_frames = 0

    def get_events(self):
        """ Logs responses/triggers """
        events = event.getKeys(timeStamped=self.session.clock)
        # waitKeys = event.waitKeys(keyList=['left','right'], timeStamped=self.session.clock)

        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?

                np.save(opj(self.session.output_dir, self.session.output_str + '_simple_response_data.npy'),
                        {'Total subject responses': self.session.total_responses})

                if self.session.settings['PRF stimulus settings']['Screenshot'] == True:
                    self.session.win.saveMovieFrames(
                        opj(self.session.screen_dir, self.session.output_str + '_Screenshot.png'))

                self.session.close()
                self.session.quit()

            for key, t in events:
                if key == self.session.mri_trigger:
                    event_type = 'pulse'
                        #marco edit. the second bit is a hack to avoid double-counting of the first t when simulating a scanner
                    if self.session.settings['PRF stimulus settings']['Scanner sync']==True:
                        self.exit_phase=True
                            #ideally, for speed, would want  getMovieFrame to be called right after the first winflip.
                            #but this would have to be dun from inside trial.run()
                    if self.session.settings['PRF stimulus settings']['Screenshot']==True:
                        self.session.win.getMovieFrame()
                else:
                    event_type = 'response'
                    self.session.total_responses += 1
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = key
                self.session.global_log.loc[idx, 'letter'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].text
                self.session.global_log.loc[idx, 'color_r'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[0]
                self.session.global_log.loc[idx, 'color_g'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[1]
                self.session.global_log.loc[idx, 'color_b'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].color[2]
                self.session.global_log.loc[idx, 'ori'] = self.session.rsvp.rsvp_stim[self.session.stim_nr].ori


                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val