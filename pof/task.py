"""

Author: Gavin Treseder
"""

# ************ Packages ********************

import math
import numpy as np
from random import random, seed
import logging
from typing import Dict, List

from dataclassy import dataclass

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.helper import flatten, str_to_dict
from pof.condition import Condition
from pof.consequence import Consequence
from pof.distribution import Distribution
import pof.demo as demo
from config import config as cf
from pof.load import Load, get_signature

# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO create set trigger and set impact method that doesn't overwrite other dicts
# TODO replace trigger with has_trigger_time

seed(1)


@dataclass(kwargs=True)
class Task(Load):
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

    # Class Variables
    CONDITION_IMPACT_AXIS = ["condition", "time"]
    CONDITION_IMPACT_METHODS = ["reduction_factor", "tbc"]
    SYSTEM_IMPACT = [None, "fm", "component", "asset"]

    # Instance variables
    name: str = "task"
    task_type: str = "factory method only"
    trigger: str = "unknown"
    active: bool = True
    cost: int = 0

    _labour = NotImplemented
    _spares = NotImplemented
    consequence = None
    _equipment = NotImplemented

    p_effective: float = 1
    triggers: Dict = {}
    impacts: Dict = {}

    t_completion: List = []
    cost_completion: List = []
    _timeline = NotImplemented

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_consequence(self.consequence)
        self.set_triggers(self.triggers)
        self.set_impacts(self.impacts)

    # ************ Load Methods **********************

    @classmethod
    def _factory(cls, task_type=None):

        if task_type == "Task":
            task_class = Task

        elif task_type == "ConditionTask":
            task_class = ConditionTask

        elif task_type == "ScheduledTask":
            task_class = ScheduledTask

        elif task_type == "Inspection":
            task_class = Inspection

        elif task_type is None:
            task_class = Task

        else:
            raise ValueError("Invalid Task Type")

        return task_class

    @classmethod
    def from_dict(cls, details=None):
        """
        Factory method for loading a Task
        """
        if isinstance(details, dict):

            task_type = details.get("task_type", None)
            task_class = cls._factory(task_type)
            task = task_class(**details)

        else:
            raise TypeError("Dictionary expected")

        return task

    @classmethod
    def demo(cls):
        return cls()

    # ************ Set Methods **********************

    def set_consequence(self, consequence=None):
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
            logging.info("Invalid Consequence")

    def set_triggers(self, triggers=None):
        if triggers is None:
            triggers = dict()
        else:
            for trigger in ["condition", "state", "time"]:
                if trigger not in triggers:
                    triggers[trigger] = dict()

            for state in triggers["state"]:
                triggers["state"][state] = bool(triggers["state"][state])

        self.triggers = triggers

    def set_impacts(self, impacts=None):
        if impacts is None:
            impacts = dict()
        else:
            for impact in ["condition", "state", "time"]:
                if impact not in impacts:
                    impacts[impact] = dict()

            if "system" not in impacts:
                impacts["system"] = []

            # Recast any ints to bools TODO make more robust
            for state in impacts["state"]:
                impacts["state"][state] = bool(impacts["state"][state])

        self.impacts = impacts

    # ************ Get Methods **********************

    def get_impacts(self):
        """Return an impact dictionary"""

        return self.impacts

    def get_triggers(self, trigger_type=None):
        """ Return a trigger dictionary"""  # TODO maybe change to object type
        if trigger_type is None:
            return self.triggers
        else:
            return self.triggers.get(trigger_type, {})

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

    def sim_completion(self, t_now, timeline=None, states=dict(), conditions=dict()):
        # TODO maybe split into states
        # TODO tasks are updating conditions that haven't been changed
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        self.record(t_now, timeline)

        # Update the condition if it was effective
        if self.is_effective(t_now, timeline):

            # Update any conditions that need to be udpated

            if "all" in self.impacts["condition"]:
                impact = self.impacts["condition"]["all"]
                for condition in conditions.values():
                    if condition.name in timeline:
                        condition.set_t_condition(t=t_now)

                    condition.reset_any(
                        target=impact["target"],
                        axis=impact["axis"],
                        method=impact["method"],
                    )
            else:
                for condition_name, impact in self.impacts["condition"].items():
                    logging.debug("Updating condition - %s" % (condition_name))

                    conditions[condition_name].set_t_condition(t=t_now)

                    conditions[condition_name].reset_any(
                        target=impact["target"],
                        axis=impact["axis"],
                        method=impact["method"],
                    )

            return self.impacts["state"]

        else:

            return dict()

    def system_impact(self):
        return self.impacts["system"]

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
        self.cost_complete = []

    # ********************* interface methods ******************

    def update_from_dict(self, dict_data):

        for key, value in dict_data.items():

            # Catch some special examples for tasks
            if key in ["trigger", "impact"]:

                for key_1, val_1 in dict_data[key].items():

                    if key_1 == "condition":
                        for key_2, val_2 in dict_data[key][key_1].items():
                            for key_3, val_3 in dict_data[key][key_1][key_2].items():
                                self.__dict__[key + "s"][key_1][key_2][key_3] = val_3

                    elif key_1 == "state":
                        for key_2, val_2 in dict_data[key][key_1].items():
                            self.__dict__[key + "s"][key_1][key_2] = val_2

                    elif key_1 == "system":
                        self.__dict__[key][key_1] = val_1
            else:
                # Use the default method
                super().update_from_dict({key: value})

    def get_dash_ids(self, prefix="", sep="-"):

        prefix = prefix + self.name + sep

        # task parameters
        param_list = ["active", "p_effective", "cost"]
        if self.trigger == "time":
            param_list = param_list + ["t_interval", "t_delay"]

        dash_ids = [prefix + param for param in param_list]

        # Triggers
        dash_ids = dash_ids + list(
            flatten(self.triggers, parent_key=prefix + "trigger", sep=sep)
        )

        # Impacts
        dash_ids = dash_ids + list(
            flatten(self.impacts, parent_key=prefix + "impact", sep=sep)
        )

        return dash_ids

    # TODO add methods for cost, resources and


class ScheduledTask(Task):  # TODO currenlty set up as emergency replacement
    """
    Parent class for creating scheduled tasks
    """

    name: str = "scheduled_task"
    trigger: str = "time"
    t_interval: int = 0
    t_delay: int = 0

    @property
    def _prop_t_interval(self):
        return self._t_interval

    @_prop_t_interval.setter
    def _prop_t_interval(self, value):

        self._t_interval = int(value)

        if math.ceil(value) != value:
            logging.warning("t_interval must be an integer - %s", value)

    @property
    def _prop_t_delay(self):
        return self._t_delay

    @_prop_t_delay.setter
    def _prop_t_delay(self, value):

        self._t_delay = int(value)

        if math.ceil(value) != value:
            logging.warning("t_interval must be an integer - %s", value)

    def sim_timeline(self, t_end, t_start=0, *args, **kwargs):

        # TODO Stubbed out to only work for trigger time and simple tile
        # TODO make it work like arange (start, stop, delay)

        if self.active:
            schedule = np.tile(
                np.linspace(self.t_interval - 1, 0, int(self.t_interval)),
                math.ceil(max((t_end - self.t_delay), 0) / self.t_interval),
            )

            if self.t_delay > 0:
                sched_start = np.linspace(self.t_delay, 0, self.t_delay + 1)
                schedule = np.concatenate((sched_start, schedule))

            schedule = np.concatenate(([schedule[0] + 1], schedule))[
                t_start : t_end + 1
            ]
        else:
            schedule = np.full(t_end - t_start + 1, -1)

        return schedule

    def update_from_dict(self, keys):

        for key, value in keys.items():

            try:
                super().update_from_dict({key: value})
            except KeyError:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise KeyError(
                        'ERROR: Cannot update "%s" - %s from dict with key %s'
                        % (self.__class__.__name__, self.name, key)
                    )

    @classmethod
    def demo(cls):
        # TODO make this a scheduled replacement task
        return cls.from_dict(demo.inspection_data["degrading"])


ScheduledTask.t_interval = ScheduledTask._prop_t_interval
ScheduledTask.t_delay = ScheduledTask._prop_t_delay


class ConditionTask(Task):
    """
    Parent class for creating condition tasks
    """

    name: str = "condition_task"
    trigger: str = "condition"
    task_completion: str = "immediate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sim_timeline(
        self, t_end, timeline, t_start=0, t_delay=NotImplemented, indicators=None
    ):
        """
        If state and condition triggers are met return the timeline met then
        """

        if self.active:

            t_end = t_end + 1
            tl_ct = np.full(t_end - t_start, True)

            # Check the state triggers have been met
            for state, trigger in self.triggers["state"].items():
                try:
                    tl_ct = (tl_ct) & (timeline[state][t_start:t_end] == trigger)
                except KeyError:
                    logging.warning("%s not found" % (state))

            # Check the condition triggers have been met
            for condition, trigger in self.triggers["condition"].items():
                try:
                    if trigger["lower"] != "min":
                        tl_ct = (tl_ct) & (
                            indicators[condition].get_timeline()[t_start:]
                            >= trigger["lower"]
                            # timeline[condition][t_start:t_end] > trigger["lower"]
                        )
                    if trigger["upper"] != "max":
                        tl_ct = (tl_ct) & (
                            indicators[condition].get_timeline()[t_start:]
                            <= trigger["upper"]
                            # timeline[condition][t_start:t_end] < trigger["upper"]
                        )
                except KeyError:
                    logging.warning("%s not found", condition)

            tl_ct = tl_ct.astype(int)

            if self.task_completion == "next_maintenance":
                # Change to days until format #Adjust
                t_lower = np.argmax(tl_ct == 1)
                t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

                tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
                tl_ct[tl_ct == False] = -1

            elif self.task_completion == "immediate":
                tl_ct = tl_ct - 1

        else:
            tl_ct = np.full(t_end - t_start, -1)

        return tl_ct

    @classmethod
    def demo(cls):
        return cls.from_dict(demo.replacement_data["on_condition"])


class Inspection(ScheduledTask):

    name: str = "inspection"
    t_interval: int = 100

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

                for trigger, threshold in self.triggers["state"].items():
                    det = det or timeline[trigger][t_now] == threshold

                # Check if any conditions are within detection threshold
                if det == True:
                    for trigger, threshold in self.triggers["condition"].items():
                        if trigger in timeline:  # TODO get more efficient check
                            if threshold["lower"] != "min":
                                det = (det) & (
                                    timeline[trigger][t_now] >= threshold["lower"]
                                )

                            if threshold["upper"] != "max":
                                det = det | (
                                    timeline[trigger][t_now] <= threshold["upper"]
                                )

        return det

    @classmethod
    def demo(cls):
        return cls.from_dict(demo.inspection_data["degrading"])


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


#     # Log it's use
#     self.t_completion = []
#     self.cost_completion = []
#     self._timeline = NotImplemented

# def __init__(
#     self,
#     name="task",
#     task_type="Factory method only",
#     trigger="unknown",
#     active=True,
#     cost=0,
#     labour=None,
#     spares=None,
#     equipment=None,
#     consequence=None,
#     p_effective=1,
#     triggers=None,
#     impacts=None,
#     *args,
#     **kwargs
# ):

#     # Task information
#     self.task_type = task_type
#     self.trigger = trigger
#     self.active = active

#     self._package = NotImplemented
#     self._impacts_parent = NotImplemented
#     self._impacts_children = False

#     # Consumed per use
#     self.cost = cost
#     self.labour = NotImplemented  # labour TODO
#     self.spares = NotImplemented  # spares TODO
#     self.equipment = NotImplemented  # equipment TODO
#     self.consequence = dict()
#     self.set_consequence(consequence)

#     # Triggers
#     self.p_effective = p_effective
#     self.set_triggers(triggers)
#     self.set_impacts(impacts)

#     # Time to execute
#     self.state = NotImplemented

#     # Log it's use
#     self.t_completion = []
#     self.cost_completion = []
#     self._timeline = NotImplemented
