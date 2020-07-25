"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from random import random, seed

from pof.condition import Condition
from pof.distribution import Distribution
from pof.consequence import Consequence

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

seed(1)

class Task:
    """
    Parameters:
                trigger
                    time, condition, state, task group?

                activty?
                    insp, repair, replace

    Things a task can do:
        - insp
            - detect symptom
            - measure condition
        - repair
            - stop initiation
            - improve condition
        - replace
            - reset everything

    """
    def __init__(self, trigger = 'time'):

        self.activity = 'any'
        self.trigger = trigger
        self.trigger_threshold_min = 0
        self.trigger_threshold_max = 10

        self.trigger_comp = '1.1.2'

        # TODO how the package is grouped together
        self.package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = 100
        self.labour = 'trade'
        self.spares = 'pole' # TODO make this an object
        self.equipment = 'ewp'
        self.consequence = Consequence()
        
        # Time to execute
        self.state = 'up' # or down

        # Log it's use
        self._count_checked = NotImplemented
        self._count_triggered = NotImplemented
        self._count_completed = NotImplemented

    def is_triggered(self):

        return

    def get_event_secheudle(self):

        # Check when it is triggered

        return NotImplemented


        




    def reset(self):
        self.triggered = False

    #TODO add methods for cost, resources and 



class Replace(Task): #TODO currenlty set up as emergency replacement

    def __init__(self):
        super().__init__()
        self.activity = 'replace' #TODO placeholder
        self.trigger = 'time' #TODO placeholder


    def is_triggered(self, failure_mode):
        """

        trigger types
        
        by state
        by time
        by condition
        by task group

        """


        if failure_mode.is_failed() == True:
            self.triggered = True
        
        return self.triggered



class Repair(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """
    
    def __init__(self, trigger=None):
        super().__init__(trigger=trigger)

        self.activty = 'repair'

        self.t_last_inspection = 0
        self.t_inspection_interval = 5
        self.t_start_inspections = 0 #TODO add this feature
        self.p_detection = 0.9

        # Repair Specific

        self.reduction_factor = 0.5
    
    def repair(self, condition):

        return NotImplemented
        




class Inspection(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """

    def __init__(self):
        super().__init__()
        self.activty = 'inspection'
        self.t_last_inspection = 0
        self.t_start_inspections = 0 #TODO add this feature
        self.p_detection = 0.9

        self.t_interval = 5

        self.schedule = dict(  #TODO not used
            start = 10,
            interval = 5,
        )

        self.triggers=dict(
            wall_thickness = dict(
                lower = 0,
                uppwer = 90,
            ),
        )

    def inspect(self, condition):
        """
        Check if the condition has been detected
        """

        #TODO rewrite this to include a measure method as opposed to simple detection

        if condition.detectable() == True:

            if random() < self.p_detection:
                return True
        
        return False

    def event_schedule(self, t_end, t_start = 0): # TODO Stubbed out to only work for trigger time and simple tile

        n_tiles = int((t_end - t_start) / self.t_interval)

        return np.tile(np.linspace(self.t_interval, 1, self.t_interval), n_tiles)

    def event_detection(self, t_end, events, current_detection = False):
        
        if current_detection == True:
            det = np.full(t_end, True)
        else:
            det = np.full(t_end,False)

            # Check if any conditions are within detection threshold
            for trigger, threshold in self.triggers.items():

                det = det | ((events[trigger] > threshold['lower']) & (trigger < threshold['upper']))

            # Check if any inspections happened

            det = (events['inspection'] == 0) & (det)

        return det

        

    # ************** Simulate **********************

    def sim_inspect(self, t_step, condition):
        
        detected = False

        # Check if an inspection occurs in the simulation step
        if self.t_last_inspection <= self.t_inspection_interval and self.t_last_inspection + t_step >= self.t_inspection_interval:
            
            # Check if anything is detected
            if self.inspect(condition) == True:
                detected = True

            self.t_last_inspection = self.t_last_inspection + t_step #TODO round it back to the inspection interval

        return detected

    def sample_event(self, df_events):
        """
        Check when detection would occur in the event table
        """

        #for trigger in triggers: # Just make it work for one now

        return NotImplemented

    # TODO
    """
        inspection -> detect failure initiation
        repair -> remove failure initiation (Failure Modes that only reset)
        restore -> remove failure initaition and restore some condition
        replace -> reset component
    """