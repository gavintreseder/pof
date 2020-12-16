"""

Author: Gavin Treseder
"""

# ************ Packages ********************

import logging
import math
from random import random, seed
from typing import List

import numpy as np
import scipy.stats as ss

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config as cf
from pof.consequence import Consequence
import pof.demo as demo
from pof.distribution import Distribution
from pof.helper import flatten, str_to_dict
from pof.pof_base import PofBase

# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO create set trigger and set impact method that doesn't overwrite other dicts
# TODO replace trigger with has_trigger_time

seed(1)

# from dataclasses import dataclass


class Task(PofBase):
    """"""

    # Class Variables
    CONDITION_IMPACT_AXIS = ["condition", "time"]
    CONDITION_IMPACT_METHODS = ["reduction_factor", "tbc"]
    SYSTEM_IMPACT = [None, "fm", "component", "asset"]
    TIME_VARIABLES = []
    POF_VARIABLES = []

    def __init__(
        self,
        name="task",
        task_type="Factory method only",
        task_category="Graphing purposes",
        task_group_name="unknown",
        trigger="unknown",
        active=True,
        cost=0,
        labour=None,
        spares=None,
        equipment=None,
        consequence=None,
        p_effective=1,
        triggers=None,
        impacts=None,
        *args,
        **kwargs,
    ):

        super().__init__(name=name, *args, **kwargs)

        # Task information
        self.task_type = task_type
        self.task_category = task_category
        self.task_group_name = task_group_name
        self.trigger = trigger
        self.active = active

        self._package = NotImplemented
        self._impacts_parent = NotImplemented
        self._impacts_children = False

        # Consumed per use
        self.cost = cost
        self.labour = NotImplemented  # labour TODO
        self.spares = NotImplemented  # spares TODO
        self.equipment = NotImplemented  # equipment TODO
        self.consequence = dict()
        self.set_consequence(consequence)

        # Triggers
        self.p_effective = p_effective
        self.set_triggers(triggers)
        self.set_impacts(impacts)

        # Time to execute
        self.state = NotImplemented

        # Log it's use
        self.t_completion = []
        self.cost_completion = []
        self._timeline = NotImplemented

    # ************ Load Methods **********************

    @classmethod
    def factory(cls, task_type=None, **kwargs):

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
    def from_dict(cls, data=None):
        """
        Factory method for loading a Task
        """
        if isinstance(data, dict):

            task_type = data.get("task_type", None)
            task_class = cls.factory(task_type)
            task = task_class(**data)

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

    # ************* sim timeline ********************

    def sim_timeline(
        self, t_end, timeline=None, t_start=0, t_delay=NotImplemented, indicators=None
    ):
        """ The common interface for all sim_timeline tasks """

        if self.active:
            timeline = self._sim_timeline(
                t_start=t_start,
                t_end=t_end,
                timeline=timeline,
                t_delay=t_delay,  # Not used
                indicators=indicators,  # Only used for condition
            )

        else:
            timeline = np.full(t_end - t_start + 1, -1)

        return timeline

    def _sim_timeline(self, *args, **kwargs):
        raise NotImplementedError()

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

    def expected(self, scaling=1):
        """ Retuns a dictionary with the quantity and cost of completing a task over time scaled by a scaling factor"""
        time, count = np.unique(self.t_completion, return_counts=True)
        quantity = count / scaling
        cost = quantity * self.cost
        return dict(active=self.active, time=time, quantity=quantity, cost=cost)

    def expected_costs(self, scaling=1):
        """ Retuns a dictionary with the cost of completing a task over time scaled by a scaling factor"""
        time, cost = np.unique(self.t_completion, return_counts=True)
        cost = cost / scaling * self.cost
        return dict(active=self.active, time=time, cost=cost)

    def expected_quantity(self, scaling=1):
        """ Retuns a dictionary with the number of times a task was completed scaled by a scaling factor"""
        time, count = np.unique(self.t_completion, return_counts=True)
        quantity = count / scaling
        return dict(time=time, quantity=quantity)

    # ********************* timeline methods ******************

    def sim_completion(self, t_now, timeline=None, states=dict(), conditions=dict()):
        # TODO maybe split into states
        # TODO tasks are updating conditions that haven't been changed
        """
        Takes a dictionary of states and dictionary of condition objects and returns the states that have been changed
        """

        self.record(t_now)

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

    def record(self, t_complete):
        """
        Record the details when a task is completed
        """
        # Time
        self.t_completion.append(t_complete)

        # Cost TODO make this variable based on time to failure
        self.cost_completion.append(self.cost)

        # TODO add other modules Resource, Labour, availability,

    # ********************* reset methods ******************

    def reset(self):
        """
        Resets the logs for a task
        """
        self.t_completion = []
        self.cost_completion = []

    # ********************* interface methods ******************

    def get_dash_ids(
        self, numericalOnly: bool, prefix: str = "", sep: str = "-", active: bool = None
    ) -> List:

        if active is None or (self.active == active):
            prefix = prefix + self.name + sep

            # task parameters
            if numericalOnly:
                param_list = ["p_effective", "cost"]
            else:
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
        else:
            dash_ids = []

        return dash_ids

    # TODO add methods for cost, resources and


class ScheduledTask(Task):  # TODO currenlty set up as emergency replacement
    """
    Parent class for creating scheduled tasks
    """

    TIME_VARIABLES = ["t_interval", "t_delay"]
    POF_VARIABLES = []

    def __init__(
        self,
        name: str = "scheduled_task",
        t_interval: int = 1,
        t_delay: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        self.trigger = "time"
        self.t_interval = t_interval
        self.t_delay = t_delay

    @property
    def t_interval(self):
        return self._t_interval

    @t_interval.setter
    def t_interval(self, value):

        if int(value) < 0:
            raise ValueError("t_interval must be a positive time - %s", value)
        else:
            self._t_interval = int(value)

            if math.ceil(value) != value:
                logging.warning("t_interval must be an integer - %s", value)

    @property
    def t_delay(self):
        return self._t_delay

    @t_delay.setter
    def t_delay(self, value):

        self._t_delay = int(value)

        if math.ceil(value) != value:
            logging.warning("t_interval must be an integer - %s", value)

    def _sim_timeline(self, t_end, t_start=0, *args, **kwargs):

        schedule = np.tile(
            np.linspace(self.t_interval, 0, int(self.t_interval) + 1),
            math.ceil(max((t_end - self.t_delay), 0) / self.t_interval),
        )

        if self.t_delay > 0:
            self.t_delay = min(self.t_delay, t_end)
            sched_start = np.linspace(self.t_delay, 0, self.t_delay + 1)
            schedule = np.concatenate((sched_start, schedule))

        schedule = schedule[t_start : t_end + 1]

        return schedule

    @classmethod
    def demo(cls):
        # TODO make this a scheduled replacement task
        return cls.from_dict(demo.inspection_data["degrading"])


class ConditionTask(Task):
    """
    Parent class for creating condition tasks
    """

    def __init__(
        self,
        name: str = "condition_task",
        task_completion: str = "immediate",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        self.trigger = "condition"
        self.task_completion = task_completion

    def _sim_timeline(
        self, t_end, timeline, t_start=0, t_delay=NotImplemented, indicators=None
    ):
        """
        If state and any condition triggers are met return the timeline met
        """

        t_end = t_end + 1
        s_trigger = np.full(t_end - t_start, True)
        c_trigger = not self.triggers["condition"]

        # Check the state triggers have been met
        for state, trigger in self.triggers["state"].items():
            s_trigger = (s_trigger) & (timeline[state][t_start:t_end] == trigger)

        # Check the condition triggers have been met
        for condition, trigger in self.triggers["condition"].items():

            tl_condition = indicators[condition].get_timeline()[t_start:]
            lower = True
            upper = True

            if (trigger["lower"] != "min") and (trigger["lower"] is not None):
                lower = tl_condition >= trigger["lower"]

            if (trigger["upper"] != "max") and (trigger["upper"] is not None):
                upper = tl_condition <= trigger["upper"]

            c_trigger = c_trigger | (lower & upper)

        tl_ct = (s_trigger & c_trigger).astype(int)

        if self.task_completion == "next_maintenance":
            # Change to days until format #Adjust
            t_lower = np.argmax(tl_ct == 1)
            t_upper = t_lower + np.argmax(tl_ct[t_lower:] == 0)

            tl_ct[t_lower:t_upper] = tl_ct[t_lower:t_upper].cumsum()[::-1] - 1
            tl_ct[tl_ct == False] = -1

        elif self.task_completion == "immediate":
            tl_ct = tl_ct - 1

        return tl_ct

    @classmethod
    def demo(cls):
        return cls.from_dict(demo.replacement_data["on_condition"])


class Inspection(ScheduledTask):
    def __init__(
        self,
        name: str = "inspection",
        t_interval: int = 100,
        t_delay: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name, t_interval=t_interval, t_delay=t_delay, *args, **kwargs
        )

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

    def effectiveness(self, pf_interval, failure_dist: Distribution = None):

        # Binomial parameters
        r = 0  # Only one succesful trial is required
        n = math.floor(pf_interval / self.t_interval)
        p = self.p_effective
        p_n = 1 - (pf_interval % self.t_interval) / self.t_interval

        # Calculate the probability of an effective inspection during the t_interval
        p_ie = p_n * (1 - ss.binom.pmf(r, n, p)) + (1 - p_n) * (
            1 - ss.binom.pmf(r, n + 1, p)
        )

        # Adjust the probability to reflect failures that occur during the t_delay
        if self.t_delay:
            if failure_dist is not None:
                p_fd = failure_dist.cdf(t_start=self.t_delay, t_end=self.t_delay)[0]
                p_ie = (1 - p_fd) * p_ie
            else:
                raise ValueError("Failure Distribution requried")

        return p_ie

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
    tsk = Task()
    print("Task - Ok")

    # # Instance variables
    # name: str = "task"
    # task_type: str = "factory method only"
    # trigger: str = "unknown"
    # active: bool = True
    # cost: int = 0

    # _labour = NotImplemented
    # _spares = NotImplemented
    # consequence = None
    # _equipment = NotImplemented

    # p_effective: float = 1
    # triggers: Dict = {}
    # impacts: Dict = {}

    # t_completion: List = []
    # cost_completion: List = []
    # _timeline = NotImplemented

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    #     self.set_consequence(self.consequence)
    #     self.set_triggers(self.triggers)
    #     self.set_impacts(self.impacts)