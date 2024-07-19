#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:02 2019

@author: marcoaqil
"""
import numpy as np
from psychopy import visual, tools
# from psychopy.visual import RadialStim
from psychopy_visionscience.radial import RadialStim
# np.random.seed(2021)

def cross_fixation(win, size=0.1, color=(1,1,1), **kwargs):
    """ Creates a fixation cross with sensible defaults. """
    return visual.ShapeStim(win, lineColor=None, fillColor=color, vertices='cross', size=size)

class FixationStim():
    """
    Small attention field task. Creates fixation color discrimination task, participant responds
    when fixation dot color changes from gray [0,0,0].
    """
    def __init__(self, session):
        self.session = session

    def draw(self, color=0, radius=0.1, outline=True):
        self.color = [color]*3 # turns into RGB array
        self.radius = radius
        if outline:
            self.stim = visual.Circle(self.session.win, lineColor=self.session.settings['fixation stim']['line_color'],
                                      lineWidth=self.session.settings['fixation stim']['line_width'],
                                      contrast=self.session.settings['fixation stim']['contrast'],
                                      radius=radius, fillColor=self.color, edges=100)
            self.stim.draw()

class AttSizeStim():
    """
    Large attention field task. Creates field for pink and blue dots, participant responds when
    proportion of dots changes from 50/50.
    """
    def __init__(self,
                 session,
                 n_sections,
                 ecc_min,
                 ecc_max,
                 n_rings,
                 row_spacing_factor,
                 opacity,
                 color1,
                 color2,
                 draw_ring=False,
                 jitter=None,
                 **kwargs):

        self.session = session
        self.n_sections = n_sections
        self.opacity = opacity
        self.draw_ring = draw_ring
        self.color1 = color1
        self.color2 = color2
        self.jitter = jitter

        total_rings = self.n_sections * (n_rings + 1) + 1

        # eccentricities for each ring of dots in degrees
        if ecc_min != 0:
            ring_eccs = np.linspace(ecc_min, ecc_max, total_rings, endpoint=True)
        else:
            ring_eccs = np.linspace(ecc_min, ecc_max, total_rings, endpoint=True)[1:] # remove centre dot if min ecc is 0

        # section positions
        section_positions = np.arange(0, total_rings, n_rings + 1)
        ring_sizes = np.diff(ring_eccs)
        blob_sizes = ring_sizes * row_spacing_factor

        circles_per_ring = ((2 * np.pi * ring_eccs[1:]) / ring_sizes).astype(int)
        n_small_jitter = circles_per_ring[0] #inner 2 rings should have smaller jitter

        element_array_np = []
        ring_nr = 1

        for ecc, cpr, s in zip(ring_eccs[:], circles_per_ring[:], blob_sizes[:]):
            if ecc in ring_eccs[:3]:
                cpr = int(cpr-3)
            if not ring_nr in section_positions:
                ring_condition = np.floor(n_sections * ring_nr / total_rings)
                for pa in np.linspace(0, 2 * np.pi, cpr, endpoint=False):
                    x, y = tools.coordinatetools.pol2cart(pa, ecc, units=None)
                    if ecc == ring_eccs[0]:
                        element_array_np.append([x,
                                            y,
                                            ecc,
                                            pa,
                                            s,
                                            1, 1, 1, 0.2, ring_nr, ring_condition])
                    else:
                        element_array_np.append([x,
                                            y,
                                            ecc,
                                            pa,
                                            s,
                                            1, 1, 1, 0.2, ring_nr, ring_condition])

            ring_nr += 1

            self.element_array_np = np.array(element_array_np)
            x_diff = np.linspace(0, np.mean(np.diff(self.element_array_np[:, 0])) / 2, 5)
            # print(f'x: {np.diff(self.element_array_np[:, 0])}\n', f'y: {np.diff(self.element_array_np[:, 1])}')

            self.element_array_stim = visual.ElementArrayStim(self.session.win,
                                                              colors=self.element_array_np[:, [5, 6, 7]],
                                                              colorSpace='rgb',
                                                              nElements=self.element_array_np.shape[0],
                                                              sizes=self.element_array_np[:, 4],
                                                              sfs=0,
                                                              opacities=self.opacity,
                                                              xys=self.element_array_np[:, [0, 1]])

        # intialize array of color orders for each trial
        n_elements =  sum(self.element_array_np[:, -1] == 0)
        if self.jitter != None:
            self.j=np.concatenate((np.random.uniform(-self.jitter/2,self.jitter/2,[self.session.n_stim,n_small_jitter,2]),\
                                np.random.uniform(-self.jitter,self.jitter,[self.session.n_stim,len(self.element_array_np)-n_small_jitter,2])),\
                                axis=1)

        self.color_orders = []
        for i in range(session.n_stim):
            i = np.arange(n_elements)
            np.random.shuffle(i)
            self.color_orders.append(i)

        self.color_orders = np.array(self.color_orders)


    def draw(self, color_balance, stim_nr,opacity):
        if self.jitter != None:
            self.element_array_stim.setXYs(self.element_array_np[:,[0,1]]+self.j[stim_nr])

        this_ring_bool = self.element_array_np[:, -1] == 0
        nr_elements_in_condition = this_ring_bool.sum()
        nr_signal_elements = int(nr_elements_in_condition * color_balance)
        ordered_signals = np.r_[np.ones((nr_signal_elements, 3)) * self.color1,
                                np.ones((nr_elements_in_condition - nr_signal_elements, 3)) * self.color2]
        ordered_signals = ordered_signals[self.color_orders][stim_nr, :]

        self.element_array_np[this_ring_bool, 5:8] = ordered_signals
        self.element_array_stim.setColors(ordered_signals, log=False)
        self.element_array_stim.setOpacities(opacity)

        self.element_array_stim.draw()

        if self.draw_ring:
            self.ring = visual.Circle(self.session.win,
                                      radius=ring_eccs[-1],
                                      lineColor=[-1, -1, -1],
                                      edges=256,
                                      opacity=0.1)
            self.ring.draw()

class TaskCue():
    """Before every trial there is a cue at fixation to indicate the task that will be performed.
    Either W for whole screen or F for fixation.
    """

    def __init__(self,session,text='W',**kwargs):
        self.session = session
        self.text = text
        self.cue = visual.TextStim(self.session.win, text=text, color=-1, height=0.1)

    def draw(self):
        self.cue.draw()

class HemiFieldStim(object):

    def __init__(self, 
                session, 
                angular_cycles, 
                angular_res,
                radial_cycles, 
                border_radius, 
                pacman_angle=20,
                n_mask_pixels=1000,
                frequency=8.0,
                eccentricity=3.0,
                width=1):
        
        self.session = session
        self.angular_cycles = angular_cycles
        self.angular_res = angular_res
        self.radial_cycles = radial_cycles
        self.border_radius = border_radius
        self.pacman_angle = pacman_angle
        self.n_mask_pixels = n_mask_pixels
        self.frequency = frequency
        self.eccentricity = eccentricity
        self.width = width

        # cosine mask, blurs the edges of the stimulus
        mask = np.ones((n_mask_pixels))
        mask[-int(border_radius*n_mask_pixels):] = 0 #(np.cos(np.linspace(0,np.pi,int(border_radius*n_mask_pixels)))+1)/2
        mask[:int(border_radius*n_mask_pixels)] = 0 #(np.cos(np.linspace(0,np.pi,int(border_radius*n_mask_pixels))[::-1])+1)/2

        self.stimulus1 = RadialStim(win=self.session.win, 
                                    mask='raisedCos', 
                                    size=((self.eccentricity+self.width)*2,(self.eccentricity+self.width)*2),#(1000.0, 1000.0), 
                                    radialCycles=self.radial_cycles, 
                                    angularCycles=self.angular_cycles, 
                                    texRes=128, 
                                    angularRes=angular_res, 
                                    visibleWedge=(0,360), 
                                    pos=(0.0, 0.0),
                                    ori=180,
                                    color=1)
        self.stimulus2 = RadialStim(win=self.session.win, 
                                    mask='raisedCos', 
                                    size=((self.eccentricity+self.width)*2,(self.eccentricity+self.width)*2), 
                                    radialCycles=self.radial_cycles, 
                                    angularCycles=self.angular_cycles, 
                                    texRes=128, 
                                    angularRes=angular_res, 
                                    visibleWedge=(0,360), 
                                    pos=(0.0, 0.0),
                                    ori=180,
                                    color=-1)

        self.inner_mask_stim = visual.Circle(self.session.win, 
                            radius=self.eccentricity, 
                            fillColor=0, 
                            edges='circle')


    def draw(self,contrast):

        stim_phase=np.mod(self.session.timer.getTime(), 1.0/self.frequency) * self.frequency
        
        if stim_phase < 0.5:      
            self.stimulus1.setContrast(contrast)   
            self.stimulus1.draw()
            self.inner_mask_stim.draw()
        else:         
            self.stimulus2.setContrast(contrast)   
            self.stimulus2.draw()
            self.inner_mask_stim.draw()

class PRFStim(object):  
    def __init__(self, session, 
                        squares_in_bar=2 ,
                        bar_width_deg=1.25,
                        tex_nr_pix=2048,
                        flicker_frequency=6, 
                        contrast=1,
                        **kwargs):
        self.session = session
        self.squares_in_bar = squares_in_bar
        self.bar_width_deg = bar_width_deg
        self.tex_nr_pix = tex_nr_pix
        self.flicker_frequency = flicker_frequency
        self.contrast = contrast

        #calculate the bar width in pixels, with respect to the texture
        self.bar_width_in_pixels = tools.monitorunittools.deg2pix(bar_width_deg, self.session.monitor)*self.tex_nr_pix/self.session.win.size[1]
        
        
        #construct basic space for textures
        bar_width_in_radians = np.pi*self.squares_in_bar
        bar_pixels_per_radian = bar_width_in_radians/self.bar_width_in_pixels
        pixels_ls = np.linspace((-self.tex_nr_pix/2)*bar_pixels_per_radian,(self.tex_nr_pix/2)*bar_pixels_per_radian,self.tex_nr_pix)

        tex_x, tex_y = np.meshgrid(pixels_ls, pixels_ls)
        
        #construct textues, alsoand making sure that also the single-square bar is centered in the middle
        if squares_in_bar==1:
            self.sqr_tex = np.sign(np.sin(tex_x-np.pi/2) * np.sin(tex_y))
            self.sqr_tex_phase_1 = np.sign(np.sin(tex_x-np.pi/2) * np.sin(tex_y+np.sign(np.sin(tex_x-np.pi/2))*np.pi/4))
            self.sqr_tex_phase_2 = np.sign(np.sign(np.abs(tex_x-np.pi/2)) * np.sin(tex_y+np.pi/2))
        else:                
            self.sqr_tex = np.sign(np.sin(tex_x) * np.sin(tex_y))   
            self.sqr_tex_phase_1 = np.sign(np.sin(tex_x) * np.sin(tex_y+np.sign(np.sin(tex_x))*np.pi/4))
            self.sqr_tex_phase_2 = np.sign(np.sign(np.abs(tex_x)) * np.sin(tex_y+np.pi/2))
            
        
        bar_start_idx=int(np.round(self.tex_nr_pix/2-self.bar_width_in_pixels/2))
        bar_end_idx=int(bar_start_idx+self.bar_width_in_pixels)+1

        self.sqr_tex[:,:bar_start_idx] = 0       
        self.sqr_tex[:,bar_end_idx:] = 0

        self.sqr_tex_phase_1[:,:bar_start_idx] = 0                   
        self.sqr_tex_phase_1[:,bar_end_idx:] = 0

        self.sqr_tex_phase_2[:,:bar_start_idx] = 0                
        self.sqr_tex_phase_2[:,bar_end_idx:] = 0
        
        
        #construct stimuli with psychopy and textures in different position/phases
        self.checkerboard_1 = visual.GratingStim(self.session.win,
                                                 tex=self.sqr_tex,
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)
        self.checkerboard_2 = visual.GratingStim(self.session.win,
                                                 tex=self.sqr_tex_phase_1,                                               
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)
        self.checkerboard_3 = visual.GratingStim(self.session.win,
                                                 tex=self.sqr_tex_phase_2,                                                
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)
        
        
        
        #for reasons of symmetry, some stimuli (4 and 8 in the order) are generated differently  if the bar has only one square
        if self.squares_in_bar!=1:                
            self.checkerboard_4 = visual.GratingStim(self.session.win,
                                                     tex=np.fliplr(self.sqr_tex_phase_1),
                                                     units='pix',
                                                     size=[self.session.win.size[1],self.session.win.size[1]],
                                                     contrast=contrast)
            self.checkerboard_8 = visual.GratingStim(self.session.win,
                                                     tex=-np.fliplr(self.sqr_tex_phase_1),
                                                     units='pix',
                                                     size=[self.session.win.size[1],self.session.win.size[1]],
                                                     contrast=contrast)
                
        else:         
            self.checkerboard_4 = visual.GratingStim(self.session.win, 
                                                     tex=np.flipud(self.sqr_tex_phase_1),
                                                     units='pix',
                                                     size=[self.session.win.size[1],self.session.win.size[1]],
                                                     contrast=contrast)
            
            self.checkerboard_8 = visual.GratingStim(self.session.win,
                                                     tex=-np.flipud(self.sqr_tex_phase_1),
                                                     units='pix',
                                                     size=[self.session.win.size[1],self.session.win.size[1]],
                                                     contrast=contrast)
        
        #all other textures are the same
        self.checkerboard_5 = visual.GratingStim(self.session.win,
                                                 tex=-self.sqr_tex,
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)
            
        self.checkerboard_6 = visual.GratingStim(self.session.win,
                                                 tex=-self.sqr_tex_phase_1,
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)
            
        self.checkerboard_7 = visual.GratingStim(self.session.win,
                                                 tex=-self.sqr_tex_phase_2,
                                                 units='pix',
                                                 size=[self.session.win.size[1],self.session.win.size[1]],
                                                 contrast=contrast)

            

        
    # this is the function that actually draws the stimulus. the sequence of different textures gives the illusion of motion.
    def draw(self, time, pos_in_ori, orientation,  bar_direction):
        
        # calculate position of the bar in relation to its orientation
        x_pos, y_pos = np.cos((2.0*np.pi)*-orientation/360.0)*pos_in_ori, np.sin((2.0*np.pi)*-orientation/360.0)*pos_in_ori
        
        # convert current time to sine/cosine to decide which texture to draw
        sin = np.sin(2*np.pi*time*self.flicker_frequency)
        cos = np.cos(2*np.pi*time*self.flicker_frequency)

        # set position, orientation, texture, and draw bar. bar moving up or down simply has reversed order of presentation
        if bar_direction==0:
            if sin > 0 and cos > 0 and cos > sin:
                self.checkerboard_1.setPos([x_pos, y_pos])
                self.checkerboard_1.setOri(orientation)
                self.checkerboard_1.draw()
            elif sin > 0 and cos > 0 and cos < sin:
                self.checkerboard_2.setPos([x_pos, y_pos])
                self.checkerboard_2.setOri(orientation)
                self.checkerboard_2.draw()
            elif sin > 0 and cos < 0 and np.abs(cos) < sin:
                self.checkerboard_3.setPos([x_pos, y_pos])
                self.checkerboard_3.setOri(orientation)
                self.checkerboard_3.draw()
            elif sin > 0 and cos < 0 and np.abs(cos) > sin:
                self.checkerboard_4.setPos([x_pos, y_pos])
                self.checkerboard_4.setOri(orientation)
                self.checkerboard_4.draw()
            elif sin < 0 and cos < 0 and cos < sin:
                self.checkerboard_5.setPos([x_pos, y_pos])
                self.checkerboard_5.setOri(orientation)
                self.checkerboard_5.draw()
            elif sin < 0 and cos < 0 and cos > sin:
                self.checkerboard_6.setPos([x_pos, y_pos])
                self.checkerboard_6.setOri(orientation)
                self.checkerboard_6.draw()
            elif sin < 0 and cos > 0 and cos < np.abs(sin):
                self.checkerboard_7.setPos([x_pos, y_pos])
                self.checkerboard_7.setOri(orientation)
                self.checkerboard_7.draw()
            else:
                self.checkerboard_8.setPos([x_pos, y_pos])
                self.checkerboard_8.setOri(orientation)
                self.checkerboard_8.draw()
        else:
            if sin > 0 and cos > 0 and cos > sin:
                self.checkerboard_8.setPos([x_pos, y_pos])
                self.checkerboard_8.setOri(orientation)
                self.checkerboard_8.draw()
            elif sin > 0 and cos > 0 and cos < sin:
                self.checkerboard_7.setPos([x_pos, y_pos])
                self.checkerboard_7.setOri(orientation)
                self.checkerboard_7.draw()
            elif sin > 0 and cos < 0 and np.abs(cos) < sin:
                self.checkerboard_6.setPos([x_pos, y_pos])
                self.checkerboard_6.setOri(orientation)
                self.checkerboard_6.draw()
            elif sin > 0 and cos < 0 and np.abs(cos) > sin:
                self.checkerboard_5.setPos([x_pos, y_pos])
                self.checkerboard_5.setOri(orientation)
                self.checkerboard_5.draw()
            elif sin < 0 and cos < 0 and cos < sin:
                self.checkerboard_4.setPos([x_pos, y_pos])
                self.checkerboard_4.setOri(orientation)
                self.checkerboard_4.draw()
            elif sin < 0 and cos < 0 and cos > sin:
                self.checkerboard_3.setPos([x_pos, y_pos])
                self.checkerboard_3.setOri(orientation)
                self.checkerboard_3.draw()
            elif sin < 0 and cos > 0 and cos < np.abs(sin):
                self.checkerboard_2.setPos([x_pos, y_pos])
                self.checkerboard_2.setOri(orientation)
                self.checkerboard_2.draw()
            else:
                self.checkerboard_1.setPos([x_pos, y_pos])
                self.checkerboard_1.setOri(orientation)
                self.checkerboard_1.draw()            
