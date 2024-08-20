"""measure your JND in orientation using a staircase method"""
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
import random
from exptools2.core import Session
import yaml
from stim import luminanceStim, AttSizeStim, FixationStim, cross_fixation
from session import PsychophysSession
from skimage import color

# load settings
settings_file=f'expsettings/stairsettings_2afc.yml'
with open(settings_file) as file:
    settings = yaml.safe_load(file)

class Stair(PsychophysSession):
    pass

ses=Stair('test-stair','logs',settings_file,eyetracker_on=False)

try:  # try to get a previous parameters file
    expInfo = fromFile('lastParams.pickle')
except:  # if not there then use a default set
    expInfo = {'observer':'jwp', 'refOrientation':0}
expInfo['dateStr'] = data.getDateStr()  # add the current time

# make a text file to save data
fileName = expInfo['observer'] #+ expInfo['dateStr']
# dataFile = open(fileName+'.csv', 'w')  # a simple text file with 'comma-separated-values'
# dataFile.write('targetSide,oriIncrement,correct\n')

# create the staircase handler
staircase = data.StairHandler(startVal = 0.05,
                          stepType = 'lin', stepSizes=0.05,
                          nUp=1, nDown=1,  # will home in on the 80% threshold
                          nTrials=50,
                          nReversals=5)

# create window and stimuli
win=ses.win
ses.create_stimuli()

hsv1=color.rgb2hsv((np.asarray(ses.settings['lum_stair']['color1'])+1)/2)
hsv2=color.rgb2hsv((np.asarray(ses.settings['lum_stair']['color2'])+1)/2)
hsv1[0]=hsv1[0]*360
hsv2[0]=hsv2[0]*360

# print(hsv1, hsv2)

blue = visual.GratingStim(win, sf=0, size=4, mask='gauss',
                          ori=expInfo['refOrientation'],
                          colorSpace='hsv',
                          color=hsv1) 
pink = visual.GratingStim(win, sf=0, size=4, mask='gauss',
                            ori=expInfo['refOrientation'],
                            colorSpace='hsv',
                            color=hsv2)
fixation = visual.GratingStim(win, color=-1, colorSpace='rgb',
                              tex=None, mask='circle', size=0.2)

# and some handy clocks to keep track of time
globalClock = core.Clock()
trialClock = core.Clock()

# display instructions and wait
message1 = visual.TextStim(win, pos=[0,+3],text='Hit a key when ready.')
message2 = visual.TextStim(win, pos=[0,-3],
    text="Then press left or right to identify which is brighter")
message1.draw()
message2.draw()
win.flip() #to show our newly drawn 'stimuli'

#pause until there's a keypress
event.waitKeys()

def end_staircase():
    # staircase has ended
    # dataFile.close()
    staircase.saveAsPickle(fileName)  # special python binary file to save all the info

    # give some output to user in the command line in the output window
    print('reversals:')
    print(staircase.reversalIntensities)
    print(f'resps{staircase.data}')

    approxThreshold = np.average(staircase.reversalIntensities[-6:])
    print('mean of final 6 reversals = %.3f' % (approxThreshold))

    # give some on-screen feedback
    feedback1 = visual.TextStim(
            win, pos=[0,+3],
            text='acc = %.3f' % (np.array(staircase.data).sum()/len(staircase.data)))

    feedback1.draw()
    # ses.fix_circle.draw()
    win.flip()
    event.waitKeys()  # wait for participant to respond

    win.close()
    core.quit()

for thisIncrement in staircase:  # will continue the staircase until it terminates!
    if staircase.thisTrialN >= staircase.nTrials:
        end_staircase()

    # set location of stimuli
    targetSide= random.choice([-1,1])  # will be either +1(right) or -1(left)
    blue.setPos([-5*targetSide, 0])
    pink.setPos([5*targetSide, 0])  # in other location

    try:
        thisResp
    except NameError:
        thisResp=1

    # set orientation of probe
    blue.setColor((hsv1[0],hsv1[1],hsv1[2] + thisIncrement*thisResp))

    # printhsv1[0],hsv1[1],hsv1[2] - thisIncrement)

    # draw all stimuli
    blue.draw()
    pink.draw()
    fixation.draw()
    win.flip()

    # wait 500ms; but use a loop of x frames for more accurate timing
    core.wait(0.5)

    # blank screen
    fixation.draw()    
    win.flip()

    # get response
    resps=[]
    thisResp=None
    while thisResp==None:
        allKeys=event.waitKeys()
        for thisKey in allKeys:
            if thisKey=='left':
                if targetSide==-1: thisResp = 1  # correct
                else: thisResp = -1              # incorrect
            elif thisKey=='right':
                if targetSide== 1: thisResp = 1  # correct
                else: thisResp = -1              # incorrect
            elif thisKey in ['q', 'escape']:
                core.quit()  # abort experiment
        event.clearEvents()  # clear other (eg mouse) events - they clog the buffer

    print(f"Trial {staircase.thisTrialN}, value: {thisIncrement:.2f}, resp: {thisResp}")
    # add the data to the staircase so it can calculate the next level
    staircase.addData(thisResp)
    # dataFile.write('%i,%.3f,%i\n' %(targetSide, thisIncrement, thisResp))
    core.wait(0.8)

end_staircase()