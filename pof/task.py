"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import math
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


        
    """def sim_timeline(self, t_end, t_start = 0): # TODO Stubbed out to only work for trigger time and simple tile

        n_tiles = int((t_end - t_start) / self.t_interval)

        return np.tile(np.linspace(self.t_interval, 1, self.t_interval), n_tiles)"""





    def reset(self):
        self.triggered = False

    #TODO add methods for cost, resources and 



class Replace(Task): #TODO currenlty set up as emergency replacement

    def __init__(self, trigger = None):
        super().__init__(trigger=trigger)
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

    def sim_timeline(self, t_end, t_start = 0):

        return NotImplemented


class Repair(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """
    
    def __init__(self, trigger=None):
        super().__init__(trigger=trigger)

        self.activty = 'repair'
        
        
        # Repair Specific
        self.reduction_factor = 0.5

        self.time_triggers = dict() #TODO maybe implement in task?
        self.state_triggers = dict()
        self.condition_triggers = dict()

        self.time_imapcts = dict() #TODO maybe implement in task?
        self.state_impacts = dict()
        self.condition_impacts = dict()
    

    def set_default(self):

        self.state_triggers = dict(
            detected = True,
        )

        self.condition_triggers = dict(
            wall_thickness = dict(
                lower = 50,
                upper = 70,
            ),

            external_diamter = dict(
                lower = 0,
                upper = 100,
            ),
        )

        self.state_impacts = dict( #True, False or N/C
            initiated = False,
            detected = False,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                reduction_factor = 0.5,
                method = 'restore',
                axis = 'condition',
            )
        )

    def sim_completion(self, states, conditions):
        """
        Takes a dictionary of states and dictionary of condition objects and returns the 
        """

        for condition_impact in self.condition_impacts.values():
            conditions[condition_impact].reset_condition(
                reduction_factor = condition_impact['reduction_factor'],
                axis = condition_impact['axis'],
                method = condition_impact['method']
            )

        self.count_completed = self.count_completed + 1

        return self.state_impacts
        
    def
    
    def sim_timeline(self, t_end, t_start = 0, t_delay = 0, conditions = dict(), states= dict()): # TODO change to trigger
        """
        If state tirgger met and condition trigger met then 
        """
        
        timeline = np.full(t_end - t_start + 1, True)
        try:
            # Check the state triggers have been met
            for state, trigger in self.state_triggers.items():
                timeline = (timeline) & (states[state])
        except KeyError:
            print("%s not found" %(state))
        
        try:
            # Check the condition triggers have been met
            for condition, trigger in self.condition_triggers.items():
                timeline = (timeline) & (conditions[condition] < trigger['upper']) & (conditions[condition] > trigger['lower'])
        except KeyError:
            print ("%s not found" %(condition))
        
            
        return timeline

class Inspection(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """

    def __init__(self, trigger=None):
        super().__init__(trigger=trigger)
        self.activity = 'inspection'
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
                upper = 90,
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

    def sim_timeline(self, t_stop, t_delay=0, t_start = 0): # TODO Stubbed out to only work for trigger time and simple tile 
        #TODO make it work like arange (start, stop, delay)
        schedule = np.tile(np.linspace(self.t_interval - 1, 0, self.t_interval), math.ceil((t_stop - t_delay) / self.t_interval))

        if t_delay > 0:
            sched_start = np.linspace(t_delay, 0, t_delay + 1)
            schedule = np.concatenate((sched_start, schedule))

        return np.concatenate(([schedule[0]+1], schedule))[t_start:t_stop+1]

    def sim_completion(self, t_end, events, current_detection = False):
        
        if current_detection == True:
            det = np.full(t_end + 1, True)
        else:
            det = np.full(t_end + 1, False)

            # Check if any conditions are within detection threshold
            for trigger, threshold in self.triggers.items():

                det = det | ((events[trigger] > threshold['lower']) & (events[trigger] < threshold['upper']))

            # Check if any inspections happened
            det = (events['inspection'] == 0) & (det)

            # Once it has been detected once, the failure mode remains detected
            det = det.cumsum().astype(np.bool)

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