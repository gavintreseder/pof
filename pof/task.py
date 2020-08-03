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

"""
Task
    - sim_timeline return a timeline of when the task is scheduled
    - sim_execution

    time based triggers
        - time
    condition based triggers
        - condition
    
    execution
        - states

"""

"""

RCM strategies

On Condition Maintenace -> reset initiation
On Condition Replacement -> reset


Predictive maintenance tasks,
Preventive Restoration or Preventive Replacement maintenance tasks,
Detective maintenance tasks,
Run-to-Failure, and
One-time changes to the "system"

"""

"""
Task
    Scheduled
        Inspection - Change detection
        Repair - Change detection and initiation
        Restoration - Change detection, initiation, failure and condition
        Replacement - Change everything
    OnCondition
        Repair - Change detection and initiation
        Restoration ...
        Repalcement


    -timeline when task can bet completed
    -timeline of dependent tasks???



    Functions for all of them
        sim_timeline() return the timeline between t_start and t_end with days remaining until task
            sim_ttr()
            sim_conditions_met()
        sim_completion() return the new states
            update conditions 
            return states after completion t=0

        sim_states() 
            update the states

    fm.update_timeline

        after completion return the timeline of state changes?
        inspection 
            -> change detection
        repair
            -> change detection (current, need new inspection to change)
            -> change failure (current, need new initiation and condition to change)
            -> change condition
        restore
            -> change detection
            -> change failure
            -> change condition
        replace
        

    scheduled -> condition

"""

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
    def __init__(self, trigger = 'any', activity = 'any'):

        self.activity = activity
        self.trigger = trigger

        self.trigger_comp = '1.1.2' #Notimplemented

        # TODO how the package is grouped together
        self._package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = 100
        self.labour = 'trade'
        self.spares = 'pole' # TODO make this an object
        self.equipment = 'ewp'
        self.consequence = Consequence()
        
        # Triggers - set in child classes

        # Impacts
        self.state_impacts = dict()
        self.condition_impacts = dict()

        # Time to execute
        self.state = 'up' # or down

        # Log it's use
        self._timeline = NotImplemented
        self._count_checked = NotImplemented
        self._count_triggered = NotImplemented
        self._count_completed = NotImplemented


    def sim_timeline(self, t_end, t_start = 0):
        """
        A function that calls an overloaded function from inherited 
        """
        if self.trigger == 'time':
            self.sim_scheduled_timeline()
        elif self.trigger == 'condition':
            self.sim_condition_timeline()

        return NotImplemented

    def sim_completion(self, timeline= None, states = dict(), conditions = dict()): #TODO maybe split into states 
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        # Update any conditions that need to be udpated
        for condition_impact in self.condition_impacts.values():
            conditions[condition_impact].reset_any(
                target = condition_impact['target'],
                reduction_factor = condition_impact['reduction_factor'],
                axis = condition_impact['axis'],
                method = condition_impact['method'],
            )

        self.count_completed = self.count_completed + 1

        return self.state_impacts

    def reset(self):
        self.triggered = False

    #TODO add methods for cost, resources and 

class ScheduledTask(Task): #TODO currenlty set up as emergency replacement

    def __init__(self, t_interval, t_delay = 0):
        super().__init__()

        self.trigger = 'time'

        self.t_delay = t_delay
        self.t_interval = t_interval


    def sim_timeline(self, t_stop, t_delay=0, t_start = 0, timeline = NotImplemented): # TODO Stubbed out to only work for trigger time and simple tile 
        #TODO make it work like arange (start, stop, delay)
        schedule = np.tile(np.linspace(self.t_interval - 1, 0, self.t_interval), math.ceil((t_stop - t_delay) / self.t_interval))

        if t_delay > 0:
            sched_start = np.linspace(t_delay, 0, t_delay + 1)
            schedule = np.concatenate((sched_start, schedule))

        return np.concatenate(([schedule[0]+1], schedule))[t_start:t_stop+1]


    def sim_completion(self):

        if current_detection == True:
            det = np.full(t_end + 1, True)
        else:
            det = np.full(t_end + 1, False)

            # Check if any conditions are within detection threshold
            for trigger, threshold in self.triggers.items():

                det = det | ((timeline[trigger] > threshold['lower']) & (timeline[trigger] < threshold['upper']))

            # Check if any inspections happened
            det = (timeline['inspection'] == 0) & (det)

            # Once it has been detected once, the failure mode remains detected
            det = det.cumsum().astype(np.bool)
        
        state = dict(
            detection = det
        )
        return self.state_impacts

class OnConditionTask(Task):

    def __init__(self, trigger = None):
        super().__init__(trigger=trigger)
        self.activity = 'replace' #TODO placeholder
        self.trigger = 'time' #TODO placeholder

        self.state_triggers = dict()
        self.condition_triggers = dict()

    def sim_timeline(self, t_end, timeline, t_start = 0, t_delay = 0): # TODO change to trigger
        """
        If state and condition triggers are met return the timeline met then 
        """
        
        tl_ct = np.full(t_end - t_start + 1, True)

        try:
            # Check the state triggers have been met
            for state, trigger in self.state_triggers.items():
                tl_ct  = (tl_ct ) & (timeline[state])
        except KeyError:
            print("%s not found" %(state))
        
        try:
            # Check the condition triggers have been met
            for condition, trigger in self.condition_triggers.items():
                tl_ct  = (tl_ct) & (timeline[condition] < trigger['upper']) & (timeline[condition] > trigger['lower'])
        except KeyError:
            print ("%s not found" %(condition))
        

        # Change to days until format #Adjust 
        tl_ct = tl_ct.astype(int)
        t_lower = np.argmax(tl_ct == 1)
        t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

        tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
        tl_ct[tl_ct == False] = -1
            
        return tl_ct

    
class ScheduledReplacement(ScheduledTask):
    
    def __init__(self, trigger = None):
        super().__init__(trigger='state')

        self.activty = 'replace'
        self.name = 'corrective_maintenance'

    def set_default(self):

        self.state_triggers = dict(
            failure = True,
        )

        self.impacts = dict(
            condition = dict(
                wall_thickness = dict(
                    target = None,
                    reduction_factor = 1,
                    method = 'restore',
                    axis = 'condition',
                ),
            ),

            state = dict(
                initiation = False,
                detection = False,
                failure = False,
            ),
        )



        self.state_impacts = dict( #True, False or N/C
            initiation = False,
            detection = False,
            failure = False,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 1,
                method = 'restore',
                axis = 'condition',
            )
        )



class OnConditionRestoration(Task):
    """
    Takes a condition (#TODO Symptoms) and determines if the failure has been detected
    """
    
    def __init__(self, trigger=None):
        super().__init__(trigger=trigger)

        self.activty = 'repair'
        self.name = 'ocr'
        # Repair Specific
        self.reduction_factor = 0.5

        self.time_triggers = dict() #TODO maybe implement in task?
        self.state_triggers = dict()
        self.condition_triggers = dict()

        self.time_imapcts = dict() #TODO maybe implement in task?
        self.state_impacts = dict()
        self.condition_impacts = dict()
    
        self.set_default()

    def set_default(self):

        self.state_triggers = dict(
            detection = True,
        )

        self.condition_triggers = dict(
            wall_thickness = dict(
                lower = 50,
                upper = 70,
            ),

            external_diameter = dict(
                lower = 0,
                upper = 100,
            ),
        )


        self.state_impacts = dict( #True, False or N/C
            initiation = False,
            detection = False,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 0.5,
                method = 'restore',
                axis = 'condition',
            )
        )

    def sim_completion(self, timeline= None, states = None, conditions = None): #TODO maybe split into states 
        """
        Takes a dictionary of states and dictionary of condition objects and returns the 
        """

        for condition_impact in self.condition_impacts.values():
            conditions[condition_impact].reset_any(
                target = condition_impact['target'],
                reduction_factor = condition_impact['reduction_factor'],
                axis = condition_impact['axis'],
                method = condition_impact['method'],
            )

        self.count_completed = self.count_completed + 1

        return self.state_impacts
        
    def sim_timeline(self, t_end, timeline, t_start = 0, t_delay = 0): # TODO change to trigger
        """
        If state tirgger met and condition trigger met then 
        """
        
        tl_ct = np.full(t_end - t_start + 1, True)

        try:
            # Check the state triggers have been met
            for state, trigger in self.state_triggers.items():
                tl_ct  = (tl_ct ) & (timeline[state])
        except KeyError:
            print("%s not found" %(state))
        
        try:
            # Check the condition triggers have been met
            for condition, trigger in self.condition_triggers.items():
                tl_ct  = (tl_ct) & (timeline[condition] < trigger['upper']) & (timeline[condition] > trigger['lower'])
        except KeyError:
            print ("%s not found" %(condition))
        

        # Change to days until format #Adjust 
        tl_ct = tl_ct.astype(int)
        t_lower = np.argmax(tl_ct == 1)
        t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

        tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
        tl_ct[tl_ct == False] = -1
            
        return tl_ct

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

    def sim_completion(self, t_end = None, timeline= None, states = None, conditions = None, current_detection = False):
        """
        Return the expected states as a result of this task
        """
        if current_detection == True:
            det = np.full(t_end + 1, True)
        else:
            det = np.full(t_end + 1, False)

            # Check if any conditions are within detection threshold
            for trigger, threshold in self.triggers.items():

                det = det | ((timeline[trigger] > threshold['lower']) & (timeline[trigger] < threshold['upper']))

            # Check if any inspections happened
            det = (timeline['inspection'] == 0) & (det)

            # Once it has been detected once, the failure mode remains detected
            det = det.cumsum().astype(np.bool)
        
        state = dict(
            detection = det
        )

        return state

        

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