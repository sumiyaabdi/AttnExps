#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:02 2019

@author: marcoaqil
"""
import numpy as np
from psychopy import visual, tools
from datetime import datetime
np.random.seed(42)

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

    def draw(self, color=0, radius=0.1):
        self.color = [color]*3 # turns into RGB array
        self.radius = radius
        self.stim = visual.Circle(self.session.win, lineColor=[0,0,0],
                                    lineWidth=self.session.settings['fixation_stim']['line_width'],
                                    contrast=self.session.settings['fixation_stim']['contrast'],
                                    radius=radius, fillColor=self.color, edges=100)
        self.stim.draw()

class rsvpStim():
    def __init__(self,
                 session):
        
        self.session = session

        if int(self.session.output_str[-1]) >= 3:
            # np.random.seed(55)
            np.random.seed(57)

        targets=np.linspace(0,self.session.n_stim,round(self.session.n_stim/self.session.settings['rsvp']['signal']),dtype=int)
        targets[0]=self.session.settings['rsvp']['jitter']+1 # avoids negative value for first target

        self.target_ids_easy=np.asarray([round(t)+round(np.random.uniform(-1,1)*self.session.settings['rsvp']['jitter']) for t in targets])
        self.target_ids_hard=np.asarray([round(t)+round(np.random.uniform(-1,1)*self.session.settings['rsvp']['jitter']) for t in targets])

        # make easy and hard targets are not the same
        overlapids=np.where(abs(self.target_ids_easy-self.target_ids_hard)<=16)
        while len(overlapids[0]) != 0:
            overlapids=np.where(abs(self.target_ids_easy-self.target_ids_hard)<=16)
            self.target_ids_easy[overlapids]+=int(self.session.settings['rsvp']['jitter']/2)
            self.target_ids_hard[overlapids]-=int(self.session.settings['rsvp']['jitter']/2)
            overlapids=np.where(abs(self.target_ids_easy-self.target_ids_hard)<=16)

        assert(self.target_ids_easy != self.target_ids_hard).all(), 'Easy and hard targets are the same!'

        self.colors=np.asarray(self.session.settings['rsvp']['colors_rgb'])
        self.ori=max(self.session.settings['rsvp']['h_target_ori'])
        self.rsvp_stim = []

        print('Easy Targets', self.target_ids_easy)
        print('Hard Targets', self.target_ids_hard)
        print('Difference bw target ids', abs(self.target_ids_easy-self.target_ids_hard))

        save_rsvp_letter=[]
        save_rsvp_ori=[]
        save_rsvp_color=[]

        for stimnr in range(self.session.n_stim):
            thisCol=self.colors[np.random.choice(len(self.colors),1)][0]

            if stimnr in self.target_ids_easy:
                thisLet=np.random.choice(self.session.settings['rsvp']['e_target'],1)[0]
                thisOri=np.random.uniform(-self.ori,self.ori)
            elif stimnr in self.target_ids_hard:
                h_id=np.random.randint(0,len(self.session.settings['rsvp']['e_target']),1)[0]
                thisLet=self.session.settings['rsvp']['h_target_letter'][h_id]
                thisOri=self.session.settings['rsvp']['h_target_ori'][h_id]
                thisCol=self.session.settings['rsvp']['h_target_color'][h_id]
            else:
                thisLet=np.random.choice(self.session.settings['rsvp']['distractors'],1)[0]
                # make sure letters don't repeat to closely to one another
                for i in range(stimnr-10,stimnr):
                    if i < 0:
                        continue
                    while (thisLet == self.rsvp_stim[i].text):
                        thisLet=np.random.choice(self.session.settings['rsvp']['distractors'],1)[0]
                if thisLet in self.session.settings['rsvp']['h_target_letter']:
                    id=list(self.session.settings['rsvp']['h_target_letter']).index(thisLet)
                    if list(thisCol)==list(self.session.settings['rsvp']['h_target_color'][id]):
                        thisOri=-self.session.settings['rsvp']['h_target_ori'][id]
                    else:
                        thisOri=np.random.uniform(-self.ori,self.ori)
                else:
                    thisOri=np.random.uniform(-self.ori,self.ori)

        
            self.rsvp_stim.append(visual.TextStim(self.session.win,
                                    text=thisLet,
                                    colorSpace='rgb',
                                    ori=round(thisOri),
                                    color=thisCol,
                                    contrast=1.2,
                                    italic=False,
                                    opacity=1,
                                    height=self.session.settings['rsvp']['letter_size'],
                                    units='deg',
                                    pos=[0, 0]))
                
            save_rsvp_letter.append(thisLet)
            save_rsvp_ori.append(thisOri)
            # save_rsvp_color.append(np.asarray(thisColor))


        if self.session.settings['rsvp']['save_stim']:
            np.save(self.session.output_dir+'/'+self.session.output_str+'_rsvp_letters'+datetime.now().strftime('%Y%m%d%H%M%S')+'.npy',\
                    save_rsvp_letter)
            np.save(self.session.output_dir+'/'+self.session.output_str+'_rsvp_ori'+datetime.now().strftime('%Y%m%d%H%M%S')+'.npy',\
                    save_rsvp_ori)
            # np.save(self.session.output_dir+'/'+self.session.output_str+'_rsvp_color'+datetime.now().strftime('%Y%m%d%H%M%S')+'.npy',\
            #         save_rsvp_color)


    def draw(self,stim_nr,target=False):

        self.rsvp_stim[stim_nr].draw()

        # if stim_nr in self.target_ids:
        #     self.rsvp_stim[stim_nr-self.session.settings['rsvp']['nback']].draw()
        # else:    
        #     self.rsvp_stim[stim_nr].draw()

    def instructions_text(self):
        # text=[]
        if self.session.difficulty == 'easy':
            text=visual.TextStim(self.session.win,
                                    text='Target: any \'X\' or \'x\'\n\n (no matter the color, orientation, or case)',
                                    colorSpace='rgb',
                                    ori=0,
                                    color='black',
                                    contrast=1.2,
                                    italic=False,
                                    opacity=1,
                                    height=2*self.session.settings['rsvp']['letter_size'],
                                    units='deg',
                                    pos=[0, 1])
            
        elif self.session.difficulty == 'hard':
            text=[]
            text.append(visual.TextStim(self.session.win,
                                    text='Targets:\n(with this specific color, orientation, and case)',
                                    colorSpace='rgb',
                                    ori=0,
                                    color='black',
                                    contrast=1.2,
                                    italic=False,
                                    opacity=1,
                                    height=2*self.session.settings['rsvp']['letter_size'],
                                    units='deg',
                                    pos=[0, 3]))
            text.append(visual.TextStim(self.session.win,
                                    text=self.session.settings['rsvp']['h_target_letter'][0],
                                    colorSpace='rgb',
                                    ori=self.session.settings['rsvp']['h_target_ori'][0],
                                    color=self.session.settings['rsvp']['h_target_color'][0],
                                    contrast=1.2,
                                    italic=False,
                                    opacity=1,
                                    height=2*self.session.settings['rsvp']['letter_size'],
                                    units='deg',
                                    pos=[-0.5, 1]))
            text.append(visual.TextStim(self.session.win,
                                    text=self.session.settings['rsvp']['h_target_letter'][1],
                                    colorSpace='rgb',
                                    ori=self.session.settings['rsvp']['h_target_ori'][1],
                                    color=self.session.settings['rsvp']['h_target_color'][1],
                                    contrast=1.2,
                                    italic=False,
                                    opacity=1,
                                    height=2*self.session.settings['rsvp']['letter_size'],
                                    units='deg',
                                    pos=[0.5, 1]))
        return text
        

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
                 pos_offset=None,
                 **kwargs):

        self.session = session
        self.n_sections = n_sections
        self.opacity = opacity
        self.draw_ring = draw_ring
        self.color1 = color1
        self.color2 = color2
        self.jitter = jitter
        self.pos_offset = pos_offset if pos_offset else 0

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

            # x_diff = np.linspace(0, np.mean(np.diff(self.element_array_np[:, 0])) / 2, 5)
            # print(f'x: {np.diff(self.element_array_np[:, 0])}\n', f'y: {np.diff(self.element_array_np[:, 1])}')
            self.element_array_np = np.array(element_array_np)
            self.element_array_np[:,0] = self.element_array_np[:,0]-self.pos_offset

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


    def draw(self, color_balance, stim_nr=None):
        if self.jitter != None:
            assert stim_nr != None, 'pass in stim_nr or remove jitter'
            self.element_array_stim.setXYs(self.element_array_np[:,[0,1]]+self.j[stim_nr])

        this_ring_bool = self.element_array_np[:, -1] == 0
        nr_elements_in_condition = this_ring_bool.sum()
        nr_signal_elements = int(nr_elements_in_condition * color_balance)
        ordered_signals = np.r_[np.ones((nr_signal_elements, 3)) * self.color1,
                                np.ones((nr_elements_in_condition - nr_signal_elements, 3)) * self.color2]
        ordered_signals = ordered_signals[self.color_orders][stim_nr, :]

        self.element_array_np[this_ring_bool, 5:8] = ordered_signals
        self.element_array_stim.setColors(ordered_signals, log=False)

        self.element_array_stim.draw()

        if self.draw_ring:
            self.ring = visual.Circle(self.session.win,
                                      radius=ring_eccs[-1],
                                      lineColor=[-1, -1, -1],
                                      edges=256,
                                      opacity=0.1)
            self.ring.draw()

class luminanceStim(AttSizeStim):
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
                pos_offset=None,
                **kwargs):
        
        super().__init__(session=session,
                        n_sections=n_sections,
                        ecc_min=ecc_min,
                        ecc_max=ecc_max,
                        n_rings=n_rings,
                        row_spacing_factor=row_spacing_factor,
                        opacity=opacity,
                        color1=color1,
                        color2=color2,
                        draw_ring=draw_ring,
                        jitter=jitter,
                        pos_offset=pos_offset)
        
    def draw(self, 
                color_balance, 
                stim_nr=None, 
                pos_offset=None):
        
        if self.jitter != None:
            assert stim_nr != None, 'pass in stim_nr or remove jitter'
            self.element_array_stim.setXYs(self.element_array_np[0,:,[0,1]]+self.j[stim_nr])

        if pos_offset:
            self.element_array_np[:,0] = self.element_array_np[:,0]-self.pos_offset
            self.element_array_stim.setXYs(self.element_array_np[:,[0,1]])

        this_ring_bool = self.element_array_np[:, -1] == 0
        nr_elements_in_condition = this_ring_bool.sum()
        nr_signal_elements = int(nr_elements_in_condition * color_balance)
        ordered_signals = np.r_[np.ones((nr_signal_elements, 3)) * self.color1,
                                np.ones((nr_elements_in_condition - nr_signal_elements, 3)) * self.color2]
        ordered_signals = ordered_signals[self.color_orders][stim_nr, :]

        self.element_array_np[this_ring_bool, 5:8] = ordered_signals
        self.element_array_stim.setColors(ordered_signals, log=False)

        self.element_array_stim.draw()

        if self.draw_ring:
            self.ring = visual.Circle(self.session.win,
                                    radius=ring_eccs[-1],
                                    lineColor=[-1, -1, -1],
                                    edges=256,
                                    opacity=0.1)
            self.ring.draw()
                    

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
