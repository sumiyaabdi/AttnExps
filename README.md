# Attention PRF Experiment
Repository for PRF mapping experiment stimulus. Created by @marcoaqil, modified and updated for the purposes of an attention field size experiment by @sumiyaabdi.

Requirements: psychopy and exptools2

**Usage**

Create setting files named expsettings_*Task*.yml within the Experiment folder. Change *Task* to your actual task name. Run the following line from within the Experient folder. 

- python main.py sub ses run E.g. `python main.py sub-001 1 1`

The command line will prompt you to enter which task (2AFC or YesNo). This will select the settings file for you.
Subject SHOULD be specified according the the BIDS convention (sub-001, sub-002 and so on), Task MUST match one of the settings files in the Experiment folder, and Run SHOULD be an integer.

## Sumiya's Notes
- Press q to quit 
- Press t to pass **Waiting for scanner** screen

# Task Details

The purpose of this experiment is to investivate differing attention size on population receptive field (pRF) maps. pRFs are mapped by using a high-contrast checkerboard bar moving in 8 cardinal directions. During this mapping participants are carring out a visual detection task that either spans the whole screen (for the large attention condition) or is the size of the fixation dot (for the small attention condition). 
