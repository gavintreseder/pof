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
    def __init__(self, trigger = 'unknown', activity = 'unknown'):

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
        self.p_effective = 1

        self.time_triggers = dict() #TODO maybe implement in task?
        self.state_triggers = dict()
        self.condition_triggers = dict()

        self.time_imapcts = dict() #TODO maybe implement in task?
        self.state_impacts = dict()
        self.condition_impacts = dict()

        # Time to execute
        self.state = 'up' # or down

        # Log it's use
        self.t_completion = []
        self._timeline = NotImplemented
        self._count_checked = NotImplemented
        self._count_triggered = NotImplemented
        self._count_completed = 0

    def set_triggers(self, all_triggers):
    
        if 'time' in all_triggers:
            self.time_triggers = all_triggers['time']
        else:
            self.time_triggers = dict()

        if 'state' in all_triggers:
            self.state_triggers = all_triggers['state']
        else:
            self.state_triggers = dict()

        if 'condition' in all_triggers:
            self.condition_triggers = all_triggers['condition']
        else:
            self.condition_triggers = dict()

    def is_effective(self, t_now=None, timeline=None):

        return random() <= self.p_effective

    def sim_completion(self, t_now, timeline= None, states = dict(), conditions = dict(), verbose = False): #TODO maybe split into states 
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        # Incur the cost of the task
        self.t_completion.append(t_now)

        # Update the condition if it was effective
        if self.is_effective(t_now, timeline):

            # Update any conditions that need to be udpated
            for condition_name, impact in self.condition_impacts.items():
                if verbose: print("updating condition - %s" %(condition_name))

                conditions[condition_name].set_condition(timeline[condition_name][t_now])

                conditions[condition_name].reset_any(
                    target = impact['target'],
                    reduction_factor = impact['reduction_factor'],
                    axis = impact['axis'],
                    method = impact['method'],
                )

            return self.state_impacts

        else:

            return dict()

        

    def reset(self):
        """
        Resets the logs for a task
        """
        self.t_completion = []

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

class ConditionTask(Task):

    def __init__(self, activity='ConditionTask'):
        super().__init__()

        self.trigger = 'condition'
        self.activity = activity

        self.task_type = 'immediate'

    def set_default(self):
        self.state_impacts = dict( #True, False or N/C
            initiation = False,
            detection = False,
            failure = False,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 0.5,
                method = 'reduction_factor',
                axis = 'condition',
            ),
            external_diameter = dict(
                target = None,
                reduction_factor = 0.5,
                method = 'reduction_factor',
                axis = 'condition',
            )
        )

        self.set_triggers(dict(
            condition= dict(
                wall_thickness = dict(
                    lower = 50,
                    upper = 70,
                )
            )
        ))

        return self


    def sim_timeline(self, t_end, timeline, t_start = 0, t_delay = NotImplemented):
        """
        If state and condition triggers are met return the timeline met then 
        """
        
        t_end = t_end + 1
        tl_ct = np.full(t_end - t_start, True)

        try:
            # Check the state triggers have been met
            for state, trigger in self.state_triggers.items():
                tl_ct  = (tl_ct ) & (timeline[state][t_start:t_end] )
        except KeyError:
            print("%s not found" %(state))
        
        try:
            # Check the condition triggers have been met
            for condition, trigger in self.condition_triggers.items():
                tl_ct  = (tl_ct) & (timeline[condition][t_start:t_end] < trigger['upper']) & (timeline[condition][t_start:t_end] > trigger['lower'])
        except KeyError:
            print ("%s not found" %(condition))
        
        tl_ct = tl_ct.astype(int)

        if self.task_type == 'next_maintenance':
            # Change to days until format #Adjust 
            t_lower = np.argmax(tl_ct == 1)
            t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

            tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
            tl_ct[tl_ct == False] = -1
        
        elif self.task_type == 'immediate':
            tl_ct = tl_ct - 1
            
        return tl_ct

class ScheduledReplacement(ScheduledTask): #Not implemented
    
    def __init__(self, t_interval, t_delay = 0):
        super().__init__(t_interval=t_interval, t_delay=t_delay)

        self.activty = 'replace'

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

        self.cost = 1000

        return self

class OnConditionRepair(ConditionTask):

    def __init__(self, activity = 'on_condition_repair'):
        super().__init__(self)

        self.activity = activity

    def set_default(self):

        self.state_impacts = dict(
            initiation = False,
            detection = False,
            failure = False,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 0,
                method = 'reduction_factor',
                axis = 'condition',
             ),
        )

        self.set_triggers(dict(
            condition= dict(
                wall_thickness = dict(
                    lower = 20,
                    upper = 80,
                )
            ),
            state = dict(
                detection = True
            )
        ))

        self.p_effective = 1

        return self



class Inspection(ScheduledTask):

    def __init__(self, t_interval, t_delay = 0):
        super().__init__(t_interval=t_interval, t_delay=t_delay)

        self.activity = 'inspection'

    def set_params(self, t_interval=None, t_delay=None, p_effective=None,state_triggers=dict(), condition_triggers=dict(), state_impacts = dict(), condition_impacts = dict()):
        
        if t_interval is not None:
            self.t_interval = t_interval
        
        if t_delay is not None:
            self.t_delay = t_delay
        
        if p_effective is not None:
            self.p_effective = p_effective
        
        if not state_triggers:
            self.state_triggers = state_triggers
        
        if not condition_triggers:
            self.condition_triggers = condition_triggers
        
        if not state_impacts:
            self.state_impacts = state_impacts

        if not condition_impacts:
            self.condition_impacts = condition_impacts

        return self

    def set_default(self):

        self.cost = 50

        self.t_interval = 5
        self.t_delay = 10

        self.p_effective = 0.9

        self.state_triggers = dict(
            initiation = True
        )

        self.condition_triggers=dict(
            wall_thickness = dict(
                lower = 0,
                upper = 90,
            ),
        )

        self.state_impacts = dict( #True, False or N/C
            detection = True,
        )

        return self

    def is_effective(self, t_now, timeline=None):
        """
        Simulate the completion of an inspection. Checks conditions and states are met
        """

        # Check if any state criteria have been met
        if timeline['detection'][t_now] == True:
            det = True
        else:
            det = False

            for trigger, threshold in self.state_triggers.items():
                det = det or timeline[trigger][t_now] == threshold

            # Check if any conditions are within detection threshold
            if det == True:
                for trigger, threshold in self.condition_triggers.items():

                    det = det | ((timeline[trigger][t_now] >= threshold['lower']) & (timeline[trigger][t_now] <= threshold['upper']))
                
        return det

class ImmediateMaintenance(ConditionTask):

    def __init__(self, activity = 'immediate_maintenance'):
        super().__init__(self)

        self.activity = activity
    
    def set_default(self):

        self.state_triggers = dict(
            failure = True,
        )

        self.condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 1,
                method = 'reduction_factor',
                axis = 'condition',
            ),
            external_diameter = dict(
                target = None,
                reduction_factor = 1,
                method = 'reduction_factor',
                axis = 'condition',
            ),
        )

        self.state_impacts = dict(
            initiation = False,
            detection = False,
            failure = False,
        )

        self.cost = 5000

        return self





#completion


# feedback
    # Describe what each section is
        # First page includes context, line of sight, performance and the gaps

    # 

    # Talk through each row line by line
    # Context

# TODO
"""
    inspection -> detect failure initiation
    repair -> remove failure initiation (Failure Modes that only reset)
    restore -> remove failure initaition and restore some condition
    replace -> reset component
"""



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