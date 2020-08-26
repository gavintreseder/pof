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

from condition import Condition
from consequence import Consequence
from distribution import Distribution

from helper import flatten


# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO create set trigger and set impact method that doesn't overwrite other dicts
# TODO replace trigger with has_trigger_time

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

    CONDITION_IMPACT_AXIS = ['condition', 'time']
    CONDITION_IMPACT_METHODS = ['reduction_factor', "tbc"]


    def __init__(self, trigger="unknown", activity="unknown", name="unknown"):

        self.name = name
        self.activity = activity
        self.trigger = trigger
        self.active = True

        self.trigger_comp = "1.1.2"  # Notimplemented

        # TODO how the package is grouped together
        self._package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = 100
        self.labour = "trade"
        self.spares = "pole"  # TODO make this an object
        self.equipment = "ewp"
        self.consequence = Consequence()

        # Triggers - set in child classes
        self.p_effective = 1

        self.triggers = dict(condition=dict(), state=dict(), time=dict())

        self.impacts = dict(condition=dict(), state=dict(), time=dict())

        self.component_reset = False

        # Time to execute
        self.state = "up"  # or down

        # Log it's use
        self.t_completion = []
        self._timeline = NotImplemented
        self._count_checked = NotImplemented
        self._count_triggered = NotImplemented
        self._count_completed = 0

    def set_triggers(self, all_triggers):

        self.triggers = all_triggers

    def get_impacts(self):
        """Return an impact dictionary"""
        
        return self.impacts

    def get_triggers(self):
        """ Return a trigger dictionary""" #TODO maybe change to object type

        return self.triggers

    def is_effective(self, t_now=None, timeline=None):

        return random() <= self.p_effective

    # ********************* expected methods ******************

    def expected_costs(self, scaling=1):
        """ Retuns a dictionary with the cost of completing a task over time scaled by a scaling factor"""
        time, cost = np.unique(self.t_completion, return_counts=True)
        cost = cost / scaling * self.cost
        return dict(time=time, cost=cost)


    def expected_counts(self, scaling=1):
        """ Retuns a dictionary with the number of times a task was completed scaled by a scaling factor"""
        time, count = np.unique(self.t_completion, return_counts=True)
        return dict(time=time, cost=count)


    # ********************* timeline methods ******************

    def sim_completion(
        self, t_now, timeline=None, states=dict(), conditions=dict(), verbose=False
    ):  # TODO maybe split into states
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        # Incur the cost of the task
        self.t_completion.append(t_now)

        # Update the condition if it was effective
        if self.is_effective(t_now, timeline):

            # Update any conditions that need to be udpated
            for condition_name, impact in self.impacts['condition'].items():
                if verbose:
                    print("Updating condition - %s" % (condition_name))

                conditions[condition_name].set_condition(
                    timeline[condition_name][t_now]
                )

                conditions[condition_name].reset_any(
                    target=impact["target"],
                    axis=impact["axis"],
                    method=impact["method"],
                )

            return self.impacts['state']

        else:

            return dict()

    def system_impact(self):
        #maybe change to impacts['system']
        return self.component_reset


    def reset(self):
        """
        Resets the logs for a task
        """
        self.t_completion = []


    def dash_update(self, dash_id, value):
        """Update the task object to a value using a dash component id"""

        try:
            ids = dash_id.split('-')

            if ids[0] in ['active', 'p_effective', 'cost', 't_interval', 't_delay']:

                if ids[0] == 'p_effective':
                    value = value / 100

                self.__dict__[ids[0]] = value

            elif ids[0] in ["trigger", "impact"]:

                if ids[1] == "condition":

                    self.__dict__[ids[0]+'s'][ids[1]][ids[2]][ids[3]] = value
                    
                elif ids[1] == 'state':
 
                    self.__dict__[ids[0]+'s'][ids[1]][ids[2]] = value
            else:
                print ("Invalid Dash ID - %s" %(dash_id))
        except:
            print ("Dash ID Error- %s" %(dash_id))

        return True

    def get_dash_ids(self, prefix='', sep='-'):

        # task parameters
        param_list = ['active', 'p_effective', 'cost']
        if self.trigger== "time":
            param_list = param_list + ['t_interval', 't_delay']

        dash_ids = [prefix + param for param in param_list]

        # Triggers
        dash_ids = dash_ids + list(flatten(self.triggers, parent_key = prefix + 'trigger', sep=sep))
        
        # Impacts
        dash_ids = dash_ids + list(flatten(self.impacts, parent_key = prefix + 'impact', sep=sep))

        return dash_ids

    # TODO add methods for cost, resources and


class ScheduledTask(Task):  # TODO currenlty set up as emergency replacement
    """
    Parent class for creating scheduled tasks
    """

    def __init__(self, t_interval, t_delay=0):
        super().__init__()

        self.trigger = "time"

        self.t_delay = t_delay
        self.t_interval = t_interval

    def set_params(
        self,
        t_interval=None,
        t_delay=None,
        p_effective=None,
        state_triggers=dict(),
        condition_triggers=dict(),
        state_impacts=dict(),
        condition_impacts=dict(),
    ):

        if t_interval is not None:
            self.t_interval = t_interval

        if t_delay is not None:
            self.t_delay = t_delay

        if p_effective is not None:
            self.p_effective = p_effective

        if not state_triggers:
            self.triggers['state'] = state_triggers

        if not condition_triggers:
            self.triggers['condition'] = condition_triggers

        if not state_impacts:
            self.impacts['state'] = state_impacts

        if not condition_impacts:
            self.impacts['condition'] = condition_impacts

        return self

    def sim_timeline(
        self, t_stop, t_delay=0, t_start=0, timeline=NotImplemented
    ):  # TODO Stubbed out to only work for trigger time and simple tile
        # TODO make it work like arange (start, stop, delay)

        if self.active:
            schedule = np.tile(
                np.linspace(self.t_interval - 1, 0, self.t_interval),
                math.ceil((t_stop - t_delay) / self.t_interval),
            )

            if t_delay > 0:
                sched_start = np.linspace(t_delay, 0, t_delay + 1)
                schedule = np.concatenate((sched_start, schedule))

            schedule = np.concatenate(([schedule[0] + 1], schedule))[
                t_start : t_stop + 1
            ]
        else:
            schedule = np.full(t_stop - t_start + 1, -1)

        return schedule


class ConditionTask(Task):
    """
    Parent class for creating condition tasks
    """

    def __init__(self, activity="ConditionTask"):
        super().__init__()

        self.trigger = "condition"
        self.activity = activity

        self.task_type = "immediate"

    def set_default(self):
          
        self.impacts = dict(
            condition = dict(
                wall_thickness=dict(
                    target=0.5,
                    method="reduction_factor",
                    axis="condition",
                ),
                external_diameter=dict(
                    target=0.5,
                    method="reduction_factor",
                    axis="condition",
                ),
            ),
            state = dict(
                initiation=False,
                detection=False,
                failure=False,
            )
        )

        self.triggers = dict(
            condition=dict(wall_thickness=dict(lower=50, upper=70,)),
            state = dict(),
        )

        return self

    def sim_timeline(self, t_end, timeline, t_start=0, t_delay=NotImplemented):
        """
        If state and condition triggers are met return the timeline met then 
        """

        if self.active:

            t_end = t_end + 1
            tl_ct = np.full(t_end - t_start, True)

            try:
                # Check the state triggers have been met
                for state, trigger in self.triggers['state'].items():
                    tl_ct = (tl_ct) & (timeline[state][t_start:t_end])
            except KeyError:
                print("%s not found" % (state))

            try:
                # Check the condition triggers have been met
                for condition, trigger in self.triggers['condition'].items():
                    tl_ct = (
                        (tl_ct)
                        & (timeline[condition][t_start:t_end] < trigger["upper"])
                        & (timeline[condition][t_start:t_end] > trigger["lower"])
                    )
            except KeyError:
                print("%s not found" % (condition))

            tl_ct = tl_ct.astype(int)

            if self.task_type == "next_maintenance":
                # Change to days until format #Adjust
                t_lower = np.argmax(tl_ct == 1)
                t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

                tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
                tl_ct[tl_ct == False] = -1

            elif self.task_type == "immediate":
                tl_ct = tl_ct - 1

        else:
            tl_ct = np.full(t_end - t_start, -1)

        return tl_ct


class ScheduledReplacement(ScheduledTask):  # Not implemented
    def __init__(self, t_interval, t_delay=0):
        super().__init__(t_interval=t_interval, t_delay=t_delay)

        self.activty = "replace"

    def set_default(self):

        self.triggers = dict(
            condition = dict(),
            state = dict(failure=True,),
        )

        self.impacts = dict(
            condition=dict(
                wall_thickness=dict(
                    target=1, method="restore", axis="condition",
                ),
            ),
            state=dict(initiation=False, detection=False, failure=False,),
        )

        self.component_reset = True

        self.cost = 1000

        return self


class OnConditionRepair(ConditionTask):
    def __init__(self, activity="on_condition_repair"):
        super().__init__(self)

        self.activity = activity

    def set_default(self):

        self.impacts = dict(
            condition = dict(
                wall_thickness=dict(
                    target=0,
                    method="reduction_factor",
                    axis="condition",
                ),
            ),
            state = dict(
                initiation=False,
                detection=False,
                failure=False,
            ),
        ) 

        self.set_triggers(
            dict(
                condition=dict(wall_thickness=dict(lower=20, upper=80,)),
                state=dict(detection=True),
            )
        )

        self.p_effective = 1

        return self


class OnConditionReplace(ConditionTask):
    def __init__(self, activity="on_condition_repair"):
        super().__init__(self)

        self.activity = activity

    def set_default(self):

       
        self.impacts = dict(
            condition = dict(
                wall_thickness=dict(
                    target=0,
                    method="reduction_factor",
                    axis="condition",
                ),
                external_diameter=dict(
                    target=0,
                    method="reduction_factor",
                    axis="condition",
                ),
            ),
            state = dict(initiation=False, detection=False, failure=False,),
        )
        self.set_triggers(
            dict(
                condition=dict(wall_thickness=dict(lower=0, upper=20,)),
                state=dict(detection=True),
            )
        )

        self.component_reset = True

        self.p_effective = 1

        return self


class Inspection(ScheduledTask):
    def __init__(self, t_interval, t_delay=0):
        super().__init__(t_interval=t_interval, t_delay=t_delay)

        self.activity = "inspection"

    def set_default(self):

        self.cost = 50

        self.t_interval = 5
        self.t_delay = 10

        self.p_effective = 0.9

        self.triggers = dict(
            condition = dict(wall_thickness=dict(lower=0, upper=90,),),
            state = dict(initiation=True)
        ) 

        self.impacts = dict(
            condition = dict(),
            state = dict(detection=True,)
        ) 

        return self

    def is_effective(self, t_now, timeline=None):
        """
        Simulate the completion of an inspection. Checks conditions and states are met
        """

        # Check if any state criteria have been met
        if timeline["detection"][t_now] == True:
            det = True
        else:
            det = False

            if random() <= self.p_effective:

                for trigger, threshold in self.triggers['state'].items():
                    det = det or timeline[trigger][t_now] == threshold

                # Check if any conditions are within detection threshold
                if det == True:
                    for trigger, threshold in self.triggers['condition'].items():

                        det = det | (
                            (timeline[trigger][t_now] >= threshold["lower"])
                            & (timeline[trigger][t_now] <= threshold["upper"])
                        )

        return det


class ImmediateMaintenance(ConditionTask):
    def __init__(self, activity="immediate_maintenance"):
        super().__init__(self)

        self.activity = activity

    def set_default(self):

        self.triggers = dict(
            condition = dict(),
            state = dict(failure=True,)
        ) 

        self.impacts = dict(
            condition= dict(
                wall_thickness=dict(
                    target=1,
                    method="reduction_factor",
                    axis="condition",
                ),
                external_diameter=dict(
                    target=1,
                    method="reduction_factor",
                    axis="condition",
                ),
            ),
            state = dict(initiation=False, detection=False, failure=False,)
        )
        
        self.component_reset = True

        self.cost = 5000

        return self


# completion


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

RCM Strat
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
        Inspection - Not Implemented
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

if __name__ == "__main__":
    task = Task()
    print("Task - Ok")
