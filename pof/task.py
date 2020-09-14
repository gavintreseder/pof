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

if __package__ is None or __package__ == '':
    from helper import flatten
    from condition import Condition
    from consequence import Consequence
    from distribution import Distribution
    import demo as demo
else:
    from pof.helper import flatten
    from pof.condition import Condition
    from pof.consequence import Consequence
    from pof.distribution import Distribution
    import pof.demo as demo

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

    def __init__(self,
                 name='task', activity='Task', trigger="unknown", active=True,
                 cost=0, labour=None, spares=None, equipment=None, consequence=None,
                 p_effective=1, triggers=dict(), impacts=dict(), component_reset=False,
                 *args, **kwargs
                 ):

        # Task information
        self.name = name
        self.activity = activity
        self.trigger = trigger
        self.active = active

        # TODO how the package is grouped together
        self._package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = cost
        self.labour = NotImplemented  # labour TODO
        self.spares = NotImplemented  # spares TODO
        self.equipment = NotImplemented  # equipment TODO
        self.set_consequence(consequence)

        # Triggers
        self.p_effective = p_effective
        self.set_triggers(triggers)
        self.set_impacts(impacts)

        self.component_reset = component_reset

        # Time to execute
        self.state = NotImplemented

        # Log it's use
        self.t_completion = []
        self.cost_completion = []
        self._timeline = NotImplemented

    # ************ Load Methods **********************

    @classmethod
    def load(cls, details=None):
        try:
            task = cls.from_dict(details)
        except:
            task = cls()
            print("Error loading %s data" % (cls.__name__))
        return task

    @classmethod
    def from_dict(cls, details=None):
        try:
            task = cls(**details)
        except:
            task = cls()
            print("Error loading %s data from dictionary" % (cls.__name__))
        return task

    # ************ Set Methods **********************

    def set_consequence(self, consequence=dict()):
        """
        Takes a Consequence object or consequence dict to set a consequence
        """

        # Load a consequence object
        if isinstance(consequence, Consequence):
            self.consequence = consequence

        # Create a consequence object
        elif isinstance(consequence, dict):
            self.consequence = Consequence(**consequence)

        elif consequence is None:
            self.consequence = Consequence()  # TODO make this a load with zero
        else:
            print("Invalid Consequence")

    def set_triggers(self, triggers=dict()):

        for trigger in ['condition', 'state', 'time']:
            if trigger not in triggers:
                triggers[trigger] = dict()

        for state in triggers['state']:
            triggers['state'][state] = bool(triggers['state'][state])

        self.triggers = triggers

    def set_impacts(self, impacts=dict()):
        for impact in ['condition', 'state', 'time']:
            if impact not in impacts:
                impacts[impact] = dict()

        # Recast any ints to bools TODO make more robust
        for state in impacts['state']:
            impacts['state'][state] = bool(impacts['state'][state])

        self.impacts = impacts

    # ************ Get Methods **********************

    def get_impacts(self):
        """Return an impact dictionary"""

        return self.impacts

    def get_triggers(self):
        """ Return a trigger dictionary"""  # TODO maybe change to object type

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
        count = count / scaling
        return dict(time=time, cost=count)

    # ********************* timeline methods ******************

    def sim_completion(
        self, t_now, timeline=None, states=dict(), conditions=dict(), verbose=False
    ):  # TODO maybe split into states
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        self.record(t_now, timeline)

        # Update the condition if it was effective
        if self.is_effective(t_now, timeline):

            # Update any conditions that need to be udpated

            if 'all' in self.impacts['condition']:
                impact = self.impacts['condition']['all']
                for condition in conditions.values():
                    condition.set_condition(
                        timeline[condition.name][t_now]
                    )

                    condition.reset_any(
                        target=impact["target"],
                        axis=impact["axis"],
                        method=impact["method"],
                    )
            else:
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
        # maybe change to impacts['system']
        return self.component_reset

    def record(self, t_start, timeline):
        """
        Record the details foth e
        """
        # Time
        self.t_completion.append(t_start)

        # Cost TODO make this variable based on time to failure
        self.cost_completion.append(self.cost)

        # TODO add other modules Resource, Labour, availability,

    # ********************* reset methods ******************

    def reset(self):
        """
        Resets the logs for a task
        """
        self.t_completion = []

    # ********************* interface methods ******************

    def update2(self, id_object, value=None):
        """

        """
        if isinstance(id_object, str):
            self.update_from_str(id_object, value, sep="-")

        elif isinstance(id_object, dict):
            self.update_from_dict(id_object)

        else:
            print("ERROR: Cannot update \"%s\" from string or dict" %
                  (self.__class__.__name__))

    def update_from_str(self, id_str, value, sep='-'):

        id_str = id_str.split(self.name + sep, 1)[1]
        id_str = id_str.split(sep)

        dict_data = {}
        for key in reversed(id_str):
            if dict_data == {}:
                dict_data = {key: value}
            else:
                dict_data = {key: dict_data}

        self.update_from_dict(dict_data)

    def dict_to_list(self, dict_data):
        keys = []
        for k, v in dict_data.items():
            if isinstance(dict_data[k], dict):
                keys.append(k)
                keys.extend(self.dict_to_list(v))
            else:
                keys.append(v)
                break
        value = keys[-1]
        keys = keys[:-1]
        return keys, value

    def update_from_dict(self, dict_data):

        keys, value = self.dict_to_list(dict_data)

        if keys[0] in ['active', 'p_effective', 'cost', 't_interval', 't_delay']:

            self.__dict__[keys[0]] = value

        elif keys[0] in ["trigger", "impact"]:

            if keys[1] == "condition":

                self.__dict__[keys[0]+'s'][keys[1]][keys[2]][keys[3]] = value

            elif keys[1] == 'state':

                self.__dict__[keys[0]+'s'][keys[1]][keys[2]] = value

        else:
            print("ERROR: Cannot update \"%s\" from dict" %
                  (self.__class__.__name__))

    def update(self, dash_id, value, sep='-'):
        """Update the task object to a value using a dash component id"""

        try:

            dash_id = dash_id.split(self.name + sep, 1)[1]

            ids = dash_id.split(sep)

            if ids[0] in ['active', 'p_effective', 'cost', 't_interval', 't_delay']:

                self.__dict__[ids[0]] = value

            elif ids[0] in ["trigger", "impact"]:

                if ids[1] == "condition":

                    self.__dict__[ids[0]+'s'][ids[1]][ids[2]][ids[3]] = value

                elif ids[1] == 'state':

                    self.__dict__[ids[0]+'s'][ids[1]][ids[2]] = value
            else:
                print("Invalid Dash ID - %s" % (dash_id))
        except:
            print("Dash ID Error- %s" % (dash_id))

    def get_dash_ids(self, prefix='', sep='-'):

        prefix = prefix + 'Task' + sep + self.name + sep

        # task parameters
        param_list = ['active', 'p_effective', 'cost']
        if self.trigger == "time":
            param_list = param_list + ['t_interval', 't_delay']

        dash_ids = [prefix + param for param in param_list]

        # Triggers
        dash_ids = dash_ids + \
            list(flatten(self.triggers, parent_key=prefix + 'trigger', sep=sep))

        # Impacts
        dash_ids = dash_ids + \
            list(flatten(self.impacts, parent_key=prefix + 'impact', sep=sep))

        return dash_ids

    # TODO add methods for cost, resources and


class ScheduledTask(Task):  # TODO currenlty set up as emergency replacement
    """
    Parent class for creating scheduled tasks
    """

    def __init__(self, t_interval=100, t_delay=0, name='scheduled_task', *args, **kwargs):  # TODO fix up defaults
        super().__init__(name=name, *args, **kwargs)

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
                np.linspace(self.t_interval - 1, 0, int(self.t_interval)),
                math.ceil((t_stop - t_delay) / self.t_interval),
            )

            if t_delay > 0:
                sched_start = np.linspace(t_delay, 0, t_delay + 1)
                schedule = np.concatenate((sched_start, schedule))

            schedule = np.concatenate(([schedule[0] + 1], schedule))[
                t_start: t_stop + 1
            ]
        else:
            schedule = np.full(t_stop - t_start + 1, -1)

        return schedule


class ConditionTask(Task):
    """
    Parent class for creating condition tasks
    """

    def __init__(self, name='condition_task', activity="ConditionTask", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.trigger = "condition"
        self.activity = activity

        self.task_type = "immediate"

    def set_default(self):

        self.impacts = dict(
            condition=dict(
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
            state=dict(
                initiation=False,
                detection=False,
                failure=False,
            )
        )

        self.triggers = dict(
            condition=dict(wall_thickness=dict(lower=50, upper=70,)),
            state=dict(),
        )

        return self

    def sim_timeline(self, t_end, timeline, t_start=0, t_delay=NotImplemented):
        """
        If state and condition triggers are met return the timeline met then
        """

        if self.active:

            t_end = t_end + 1
            tl_ct = np.full(t_end - t_start, True)

            # Check the state triggers have been met
            for state, trigger in self.triggers['state'].items():
                try:
                    tl_ct = (tl_ct) & (
                        timeline[state][t_start:t_end] == trigger)
                except KeyError:
                    print("%s not found" % (state))

            # Check the condition triggers have been met
            for condition, trigger in self.triggers['condition'].items():
                try:
                    if trigger['lower'] != 'min':
                        tl_ct = (
                            (tl_ct)
                            & (timeline[condition][t_start:t_end] > trigger["lower"])
                        )
                    if trigger['upper'] != 'max':
                        tl_ct = (
                            (tl_ct)
                            & (timeline[condition][t_start:t_end] < trigger["upper"])
                        )
                except KeyError:
                    print("%s not found" % (condition))

            tl_ct = tl_ct.astype(int)

            if self.task_type == "next_maintenance":
                # Change to days until format #Adjust
                t_lower = np.argmax(tl_ct == 1)
                t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

                tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[
                    ::-1] - 1
                tl_ct[tl_ct == False] = -1

            elif self.task_type == "immediate":
                tl_ct = tl_ct - 1

        else:
            tl_ct = np.full(t_end - t_start, -1)

        return tl_ct


class ScheduledReplacement(ScheduledTask):  # Not implemented
    def __init__(self, t_interval, t_delay=0, name='scheduled_replacement'):
        super().__init__(t_interval=t_interval, t_delay=t_delay)

        self.name = name
        self.activty = "replace"

    def set_default(self):

        self.triggers = dict(
            condition=dict(),
            state=dict(failure=True,),
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
    def __init__(self, activity="on_condition_repair", name='on_condition_repair'):
        super().__init__(self)

        self.name = name
        self.activity = activity

    def set_default(self):

        self.impacts = dict(
            condition=dict(
                wall_thickness=dict(
                    target=0,
                    method="reduction_factor",
                    axis="condition",
                ),
            ),
            state=dict(
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


class OnConditionReplacement(ConditionTask):
    def __init__(self, activity="on_condition_replace", name='on_condition_replace'):
        super().__init__(self)

        self.name = name
        self.activity = activity

    def set_default(self):

        self.impacts = dict(
            condition=dict(
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
            state=dict(initiation=False, detection=False, failure=False,),
        )
        self.set_triggers(
            dict(
                condition=dict(wall_thickness=dict(lower=0, upper=20,),
                               external_diameter=dict(lower=0, upper=20,)),
                state=dict(detection=True),
            )
        )

        self.component_reset = True

        self.p_effective = 1

        return self


class Inspection(ScheduledTask):

    def __init__(self, t_interval=100, t_delay=0, name='inspection',  *args, **kwargs): #TODO fix up the defaults
        
        super().__init__(t_interval=t_interval, t_delay=t_delay, *args, **kwargs)

        self.name = name
        self.activity = "Inspection"

    def set_default(self):

        self.cost = 50

        self.t_interval = 5
        self.t_delay = 10

        self.p_effective = 0.9

        self.triggers = dict(
            condition=dict(wall_thickness=dict(lower=0, upper=90,),),
            state=dict(initiation=True)
        )

        self.impacts = dict(
            condition=dict(),
            state=dict(detection=True,)
        )

        return self

    # TODO replace is_effective with trigger check
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
    def __init__(self, activity="immediate_maintenance", name='immediate_maintenance'):
        super().__init__(self)

        self.name = name
        self.activity = activity

    def set_default(self):

        self.triggers = dict(
            condition=dict(),
            state=dict(failure=True,)
        )

        self.impacts = dict(
            condition=dict(
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
            state=dict(initiation=False, detection=False, failure=False,)
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
        Repair - Change detection and initiation (as bad as old)
        Restoration - Change detection, initiation, failure and condition (RF 0 < 1)
        Replacement - Change everything
    OnCondition
        Inspection - Not Implemented
        Repair - Change detection and initiation
        Restoration ...
        Replacement


    # Trigger
    # Impact
    # Schedule -> can it be completed now, next maintenane or does it need an immediate action
        # Planned
        # Reactive
    # Level of Failure


    # Trigger
        # Time
        # State
        # Condition
    # Impact
        # Time
        # State
        # Condition
    # Level of Impact
        # Nil
        # Failure Mode
        # Component




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
