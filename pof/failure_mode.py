"""

Author: Gavin Treseder
"""

# ************ Packages ********************

import configparser
import copy
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from random import random, seed

import numpy as np
import pandas as pd
import scipy.stats as ss
from collections.abc import Iterable
from scipy.linalg import circulant
from matplotlib import pyplot as plt
from tqdm import tqdm
from lifelines import WeibullFitter

# Change the system path when this is run directly
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.helper import fill_blanks, id_update, str_to_dict
from pof.indicator import Indicator, ConditionIndicator
from pof.distribution import Distribution
from pof.consequence import Consequence
from pof.task import (
    Task,
    Inspection,
    ConditionTask,
    OnConditionRepair,
    OnConditionReplacement,
    ImmediateMaintenance,
)
import pof.demo as demo
from pof.load import Load
from config import config as cf

# TODO
"""
    - Use condition pf to change indicator 
    - task indicator
"""

# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO Change this to update timeline based on states that have changed
# TODO make it work with non zero start times
# TODO add method for multiple defaults

cf = cf["FailureMode"]

seed(1)

PF_CURVES = ["linear", "step"]
REQUIRED_STATES = ["initiation", "detection", "failure"]


@dataclass(repr=False)
class FailureModeData(Load):
    """
    A class that contains the data for the FailureMode object.

    This is a temporary fix due to issues with @property and @dataclass changing the constructor to only accept data with leading underscores
    """

    # pylint: disable=too-many-instance-attributes
    # Reasonable in this implementation

    name: str = cf.get("name")
    active: bool = cf.getboolean("active")
    pf_curve: str = cf.get("pf_curve")
    pf_interval: int = cf.get("pf_interval")
    pf_std: int = cf.get("pf_std")

    # Set methods used
    dists: Dict = None
    consequences: Dict = None
    indicators: Dict = None
    conditions: Dict = None
    states: Dict = None
    tasks: Dict = None
    init_states: Dict = None

    # Simulation Details
    timeline: Dict = None
    timelines: Dict = None
    sim_counter: Dict = None


class FailureMode(FailureModeData):

    """
    A class that contains the methods for FailureMode object

    Params:
        see FailureModeData

    Methods:
        Overloaded methods for properties with set logic

    Usage:

    First import FailureMode from the failure_mode module

        >>> from failure_mode import FailureMode

    Create a FailureMode object

        >>> FailureMode()
        <failure_mode.FailureMode object at 0x...>

    """  # TODO check why failure_mode is included

    # *************** Property Methods *************
    # #TODO convert any get/set pairs to properties

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def pf_curve(self):
        return self._pf_curve

    @pf_curve.setter
    def pf_curve(self, value):
        """ Set the pf_curve to a valid str"""

        if value in PF_CURVES:
            self._pf_curve = value
        else:
            raise ValueError(
                "%s (%s) - pf_curve must be from: %s"
                % (self.__class__.__name__, self.name, PF_CURVES)
            )

    @property
    def pf_interval(self):
        return self._pf_interval

    @pf_interval.setter
    def pf_interval(self, value):

        try:
            if value > 0:
                self._pf_interval = value
            else:
                raise ValueError(
                    "%s (%s) - pf_interval must be greater than 0"
                    % (self.__class__.__name__, self.name)
                )
        except:
            raise

    @property
    def dists(self):
        """
        Usage:

        A single distribution can be added to the dists dictionary

            >>> fm.dists = Distribution(name='untreated')
            >>> fm.dists
            {'untreated': <pof.distribution.Distribution object at 0x...

            >>> fm.dists = dict(name='dist_from_dict', alpha=50, beta = 10, gamma=1)
            >>> fm.dists
            {'dist_from_dict': <pof.distribution.Distribution object at 0x..

        An iterator of distributions can be added to the dists dictionary

            >>> fm.dists = dict('fm_name' = Distribution())
            >>> fm.dists
            {'fm_name': <pof.distribution.Distribution object at 0x...

            >>> fm.dists = dict('first_dist' = dict(name='first_dist', alpha=50, beta=10, gamma=1))
            {'first_dist': <pof.distribution.Distribution object at 0x...

        """
        return self._dists

    @dists.setter
    def dists(self, value):

        # TODO maybe just update init each time anyway?
        untreated = copy.copy(getattr(getattr(self, "_dists", None), "untreated", None))

        self._set_container_attr("_dists", Distribution, value)

        # Check if 'untreated' was updated and if so, call init dist
        if untreated != self.dists.get("untreated", None):
            self.set_init_dist()

    def dists2(self, value):

        # TODO Illyse -> see if this logic works for other containers
        """ Set the distribution"""

        # Create an empty dictionary if it doesn't exist #Dodgy fix because @property error
        if getattr(self, "_dists", None) is None:
            self._dists = dict()

        try:
            # Add the value to the dictionary if it is a Distribution
            if isinstance(value, Distribution):
                self._dists[value.name] = value

            # Check if the input is an iterable
            elif isinstance(value, Iterable):

                # Create a
                # Is this dict a valid source for creating the ojbect

                if "name" in value:  # TODO Check all keys in function
                    self._dists[value["name"]] = Distribution.load(value)

                else:
                    # Iterate through and create objects
                    for val in value.values():
                        # Calls this method again with the inside value
                        self.dists = val

            else:
                raise ValueError

        except:
            if value is None and cf.USE_DEFAULT is True:
                print(
                    "%s (%s) - Distribution cannot be set from %s - Default Uses"
                    % (self.__class__.__name__, self.name, value)
                )
            else:
                raise ValueError(
                    "%s (%s) - Distribution cannot be set from %s"
                    % (self.__class__.__name__, self.name, value)
                )

    @property
    def timeline(self):
        return self._timeline

    @timeline.setter
    def timeline(self, value):
        self._timeline = value

    @property
    def timelines(self):
        return self._timelines

    @timelines.setter
    def timelines(self, value):
        self._timelines = value

    # ************** Set Functions *****************

    def set_consequence(self, consequence):

        self.cof = Consequence()

    def set_conditions(self, input_condition=None):
        """
        Takes a dictionary of conditions and sets the failure mode conditions
        """
        if input_condition is None:
            if cf.FILL_NONE_WITH_DEFAULT == True:
                # Create a default condition using failure mode parameters
                print(
                    "Failure Mode (%s) - No condition provided - Default condition created"
                    % (self.name)
                )
                self.conditions[None] = ConditionIndicator(
                    pf_curve=self.pf_curve,
                    pf_interval=self.pf_interval,
                    pf_std=self.pf_std,
                )
            else:
                raise ValueError(
                    "Failure Mode (%s) - No condition provided" % (self.name)
                )
        else:
            for cond_name, condition in input_condition.items():

                # Load a condition object
                if isinstance(condition, Indicator):
                    self.conditions[cond_name] = condition

                # Add a name to the distribution and set create the object
                elif isinstance(condition, dict):

                    # if self.conditions is not None:
                    #     if isinstance(self.conditions[cond_name], Indicator):
                    #         self.conditions[cond_name].update_from_dict(condition)
                    #     else:
                    #         for key, value in condition.items():
                    #             self.conditions[key] = value

                    # else:
                    self.conditions[cond_name] = ConditionIndicator.from_dict(condition)

                else:
                    print(
                        'ERROR: Cannot update "%s" condition from dict'
                        % (self.__class__.__name__)
                    )

    def set_untreated(self, untreated):
        """
        Takes a Distribution, a dictionary that represents a Distribution or an iterable of these objects and sets untreated to those objects
        Usage


        set_untreated(Distribution)
        self.untreated == Distribution

        set_untreated(dict(name = Distribution))
        self.untreated == Distribution

        set_untreated(dict("untreated" = dict("alpha" = 10)))
        self.untreated.alpha == 10

        set_untreated(dict_data(untreated))
        self.untreated == Distribution
        """
        # Load a distribution object
        if isinstance(untreated, Distribution):
            self.untreated = untreated

        # Add a name to the distribution and set create the object
        elif isinstance(untreated, dict):
            # is it a dist in a dict

            # does it already exist
            # if untreat.name is in self.untreated:
            # if yes update
            # if no create

            # TODO Illyse, was this commented out block needed?
            """
            if self.untreated is not None:
                self.untreated.update_from_dict(untreated)
            else:
                untreated["name"] = "untreated"
                self.untreated = Distribution.from_dict(untreated)
            """

            untreated["name"] = "untreated"
            self.untreated = Distribution.from_dict(untreated)

        else:
            untreated["name"] = "untreated"
            self.untreated = Distribution.from_dict(untreated)
            # print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

        # Set the probability of initiation using the untreated parameters
        self.set_init_dist()

    def set_init_states(self, states):
        # TODO Update this method at the same time as set state

        if self.init_states is None:
            if cf.USE_DEFAULT:
                self.init_states = {state: False for state in REQUIRED_STATES}
            else:
                raise ValueError("Failure Mode - %s - No states provided" % (self.name))

        # update from the input argument
        if states is not None:
            self.init_states.update(states)

        # Update from required states
        for state in REQUIRED_STATES:
            if state not in self.init_states:
                self.init_states[state] = False

    def set_states(self, states=None):
        # TODO check this on the wekeend and split into set and update methods. Set at start. Update
        # TODO make this work with actual asset
        # TODO this is super super ugly. make state an indiciator or it's own class.

        # Set a default value if none has been provided
        if self.states is None:
            if cf.USE_DEFAULT:
                self.states = {state: False for state in REQUIRED_STATES}
            else:
                raise ValueError("Failure Mode - %s - No states provided" % (self.name))

        # update from the input argument
        if states is not None:
            self.states.update(states)

        # Update from required states
        for state in REQUIRED_STATES:
            if state not in self.states:
                self.states[state] = False

    def set_tasks2(self, tasks):
        """
        Takes a dictionary of tasks and sets the failure mode tasks
        """

        for task_name, task in tasks.items():
            if isinstance(task, Task):
                self.tasks[task_name] = task
            elif isinstance(task, dict):
                if task["activity"] == "Inspection":
                    self.tasks[task["name"]] = Inspection.load(task)
                elif task["activity"] == "ConditionTask":
                    self.tasks[task_name] = ConditionTask.load(task)

                else:
                    print("Invalid Task Activity")
            else:
                print("Invalid Task")

    def set_tasks(self, tasks):
        """
        Takes a dictionary of tasks and sets the failure mode tasks
        """
        print(tasks)
        # Check if Task
        if isinstance(tasks, Task):
            self.tasks[tasks.name] = tasks

        # Check if Dictionary
        elif isinstance(tasks, dict):
            # Iterate through Dictionary
            for task_name, task in tasks.items():
                print(task_name)
                # is it a Task
                if isinstance(task, Task):
                    self.tasks[task_name] = task
                # does it exist: yes, update
                elif task_name in self.tasks:
                    if isinstance(self.tasks[task_name], dict):
                        if task["activity"] == "Inspection":
                            self.tasks[task["name"]] = Inspection.load(task)
                        elif task["activity"] == "ConditionTask":
                            self.tasks[task_name] = ConditionTask.load(task)
                    # update method
                    else:
                        self.tasks[task_name].update_from_dict(task)
                # does it exist: no, create from dictionary
                elif isinstance(task, dict):
                    if task["activity"] == "Inspection":
                        self.tasks[task["name"]] = Inspection.load(task)
                    elif task["activity"] == "ConditionTask":
                        self.tasks[task_name] = ConditionTask.load(task)
                    else:
                        print("Invalid Task Activity")
                else:
                    self.tasks[tasks["name"]].update_from_dict(tasks)
                    break
        else:
            print("Invalid Task")

    def link_indicator(self, indicator):
        """
        Takes an indicator ojbect, or an iterable list of objects and links condition
        """

        # Link all the indicators so they can be used for tasks
        self.indicator = (
            indicator  # TODO eventually replace all conditon with indicator
        )

        # If it is an iterable, link them all
        if isinstance(indicator, Iterable):
            for indicator in indicator.values():
                if indicator.name in self.conditions:
                    self.conditions[indicator.name] = indicator

        # If there is only one update
        else:
            if indicator.name in self.conditions:
                self.conditions[indicator.name] = indicator

    # ************** Get Functions *****************

    def get_states(self):
        return self.states

    def sf(self, t_start, t_end):

        # TODO add other methods here
        if self.pof is not None:
            return self.pof.sf(t_start, t_end)

    # ************** Is Function *******************

    def is_failed(self):
        return self.states["failure"]

    def is_initiated(self):
        return self.states["initiation"]

    def is_detected(self):
        return self.states["detection"]

    # *************** Get methods *******************

    def set_init_dist(self):  # TODO needs to get passed a condition and a pof
        """
        Convert the probability of failure into a probability of initiation
        """
        # TODO being moved to Distribution

        # Super simple placeholder # TODO add other methods
        alpha = self.dists["untreated"].alpha
        beta = self.dists["untreated"].beta
        if self.pf_interval is None:
            gamma = max(0, self.dists["untreated"].gamma)
        else:
            gamma = max(0, self.dists["untreated"].gamma - self.pf_interval)

        # TODO add an adjustment to make sure the pfinterval results in a resaonable gamma
        # self.pf_interval = self.pf_interval - max(self.gamma - self.pf_interval + self.pf_std * 3)

        self.dists["init"] = Distribution(
            alpha=alpha, beta=beta, gamma=gamma, name="init"
        )

    def get_expected_pof(self):

        # TODO add a check if it has been simulated yet self.pof is None, self._timlines = None

        return self.pof

    # ****************** Timeline ******************

    def mc_timeline(self, t_end, t_start=0, n_iterations=100):

        self.reset()  # TODO ditch this

        for i in tqdm(range(n_iterations)):
            self.sim_timeline(t_end=t_end, t_start=t_start)
            self.save_timeline(i)

    def sim_timeline(self, t_end, t_start=0, verbose=False):

        timeline = self.init_timeline(t_start=t_start, t_end=t_end)

        if self.active:

            t_now = t_start

            while t_now < t_end:

                # Check when the next task needs to be executed
                t_now, task_names = self.next_tasks(timeline, t_now, t_end)

                # Complete those tasks
                self.complete_tasks(t_now, task_names, verbose=verbose)

                t_now = t_now + 1

            self.increment_counter()
            self.reset_for_next_sim()

        return self.timeline

    def complete_tasks(self, t_now, task_names, verbose=False):
        """ Executes the tasks """

        system_impacts = []
        if self.active:
            for task_name in task_names:
                if verbose:
                    print(t_now, task_names)

                # Complete the tasks
                states = self.tasks[task_name].sim_completion(
                    t_now,
                    timeline=self.timeline,
                    states=self.get_states(),
                    conditions=self.indicator,
                    verbose=verbose,
                )

                # Update timeline
                self.set_states(states)
                self.update_timeline(t_now + 1, updates=states, verbose=verbose)

                # Check if a system impact is triggered
                system_impacts.append(self.tasks[task_name].system_impact())

        return system_impacts

    def init_timeline(self, t_end, t_start=0):

        if self.active:
            self._init_timeline(t_end, t_start)

        else:
            self.timeline = dict(
                time=np.linspace(t_start, t_end, t_end - t_start + 1, dtype=int),
                intiation=np.full(t_end + 1, False),
                dectection=np.full(t_end + 1, False),
                failure=np.full(t_end + 1, False),
            )

        return self.timeline

    def _init_timeline(self, t_end, t_start=0):
        """
        Simulates a single timeline to determine the state, condition and tasks
        """

        # Create a timeline
        timeline = dict(
            time=np.linspace(t_start, t_end, t_end - t_start + 1, dtype=int)
        )

        # Get intiaition
        timeline["initiation"] = np.full(t_end + 1, self.is_initiated())
        t_initiate = 0
        if not self.is_initiated():
            # TODO this needs to be conditional_sf
            t_initiate = min(t_end + 1, int(self.init_dist.sample()))
            timeline["initiation"][t_initiate:] = 1

        # Get condition
        for condition_name, condition in self.conditions.items():
            timeline[condition_name] = condition.sim_timeline(
                t_start=t_start - t_initiate,
                t_stop=t_end - t_initiate,
                pf_interval=self.pf_interval,
                pf_std=self.pf_std,
            )

        # Get the indicators
        for indicator in self.indicator.values():
            if indicator.name not in self.conditions:
                timeline[indicator.name] = indicator.get_timeline()

        # Check failure
        timeline["failure"] = np.full(t_end + 1, self.is_failed())
        if not self.is_failed():
            for condition in self.conditions.values():
                tl_f = condition.sim_failure_timeline(
                    t_start=t_start - t_initiate,
                    t_stop=t_end - t_initiate,
                    pf_interval=self.pf_interval,
                    pf_std=self.pf_std,
                )
                timeline["failure"] = (timeline["failure"]) | (tl_f)

        # Check tasks with time based trigger
        for task in self.tasks.values():

            if task.trigger == "time":
                timeline[task.name] = task.sim_timeline(t_end)

        # Initialised detection
        timeline["detection"] = np.full(t_end - t_start + 1, self.is_detected())

        # Check tasks with condition based trigger
        for task_name, task in self.tasks.items():

            if task.trigger == "condition":
                timeline[task_name] = task.sim_timeline(t_end, timeline)

        self.timeline = timeline

        return timeline

    def update_timeline(self, t_start, t_end=None, updates=dict(), verbose=False):
        """
        Takes a timeline and updates tasks that are impacted
        """

        if t_end is None:
            t_end = self.timeline["time"][-1]

        # Initiation -> Condition -> time_tasks -> states -> tasks
        if t_start < t_end:

            if "time" in updates:
                self.timeline["time"] = np.linspace(
                    t_start, t_end, t_end - t_start + 1, dtype=int
                )

            # Check for initiation changes
            if "initiation" in updates:
                t_initiate = min(
                    # TODO this needs to be condiitonal sf
                    t_end,
                    t_start + int(self.init_dist.sample()),
                )  # TODO make this conditional
                self.timeline["initiation"][t_start:t_initiate] = updates["initiation"]
                self.timeline["initiation"][t_initiate:] = True
            else:
                t_initiate = np.argmax(self.timeline["initiation"][t_start:] > 0)

            # Check for condition changes
            for condition_name, condition in self.conditions.items():
                if "initiation" in updates or condition_name in updates:
                    if verbose:
                        print(
                            "condition %s, start %s, initiate %s, end %s"
                            % (condition_name, t_start, t_initiate, t_end)
                        )
                    # self.conditions[condition_name].set_condition(self.timeline[condition_name][t_start]) #TODO this should be set earlier using a a better method
                    self.timeline[condition_name][t_start:] = condition.sim_timeline(
                        t_start=t_start - t_initiate,
                        t_stop=t_end - t_initiate,
                        pf_interval=self.pf_interval,
                        pf_std=self.pf_std,
                    )

            # Check for detection changes
            if "detection" in updates:
                self.timeline["detection"][t_start:] = updates["detection"]

            # Check for failure changes
            if "failure" in updates:
                self.timeline["failure"][t_start:] = updates["failure"]
                for condition in self.conditions.values():
                    tl_f = condition.sim_failure_timeline(
                        t_start=t_start - t_initiate, t_stop=t_end - t_initiate
                    )
                    self.timeline["failure"][t_start:] = (
                        self.timeline["failure"][t_start:]
                    ) | (tl_f)

            # Check for new task timelines
            for task_name, task in self.tasks.items():

                # Update time based tasks
                if task.trigger == "time" and task_name in updates:
                    self.timeline[task_name][t_start:] = task.sim_timeline(
                        s_tart=t_start, t_end=t_end, timeline=self.timeline
                    )

                # Update condition based tasks if the failure mode initiation has changed
                if task.trigger == "condition":
                    self.timeline[task_name][t_start:] = task.sim_timeline(
                        t_start=t_start, t_end=t_end, timeline=self.timeline
                    )

        return self.timeline

    # def _update_indicators(self, t_start, t_end=None):

    def next_tasks(self, timeline=None, t_start=0, t_end=None):
        """
        Takes a timeline and returns the next time, and next task that will be completed
        """

        if timeline is None:
            timeline = self.timeline

        if t_end is None:
            t_end = timeline["time"][-1]

        next_tasks = []
        next_time = t_end

        # TODO make this more efficient by using a task array rather than a for loop
        if self.active:
            for task in self.tasks:

                if 0 in timeline[task][t_start:]:
                    t_task = (
                        timeline["time"][np.argmax(timeline[task][t_start:] == 0)]
                        + t_start
                    )

                    if t_task < next_time:
                        next_tasks = [task]
                        next_time = t_task
                    elif t_task == next_time:
                        next_tasks = np.append(next_tasks, task)

        return next_time, next_tasks

    def save_timeline(self, i):
        self._timelines[i] = self.timeline

    def increment_counter(self):
        self._sim_counter = self._sim_counter + 1

    # ****************** Expected Methods  ************

    def expected_simple(self):
        """Returns all expected outcomes using a simple average formula"""

        # TODO strip out the values that don't matter

        self.expected = dict()
        self.uncertainty = dict()
        self.lower = dict()
        self.upper = dict()

        for key in self._timelines[0]:

            all_values = np.array([self._timelines[d][key] for d in self._timelines])

            self.expected[key] = all_values.mean(axis=0)
            self.uncertainty[key] = all_values.std(axis=0)
            self.lower[key] = np.percentile(all_values, 10, axis=0)
            self.upper[key] = np.percentile(all_values, 90, axis=0)

        return self.expected

    def expected_pof(self):
        # TODO general into expected event = 'failure', cumulative = True/False method
        t_failures = []

        t_max = self._timelines[0]["time"][-1] + 1

        # Get the time of first failure or age at failure
        for timeline in self._timelines.values():
            if timeline["failure"].any():
                t_failures.append(timeline["time"][timeline["failure"]][0])
            else:
                t_failures.append(t_max)

        # Fit the weibull
        wbf = WeibullFitter()

        event_observed = t_failures != t_max

        wbf.fit(durations=t_failures, event_observed=event_observed)

        self.pof = Distribution(
            alpha=wbf.lambda_,
            beta=wbf.rho_,
        )

        return self.pof

    def _expected(self, timeline_key, first_event=True):
        # TODO general into expected event = 'failure', cumulative = True/False method
        # TODO delete
        t_events = []

        t_min = self._timelines[0]["time"][0]
        t_max = self._timelines[0]["time"][-1] + 1

        # Get the time of first failure or age at failure
        for timeline in self._timelines.values():
            if timeline["failure"].any():
                t_event = timeline["time"][timeline["failure"]]
                if first_event:
                    t_events.append(t_event[0])
                else:
                    t_events.append(np.diff(np.append(t_min, t_event)))
            else:
                t_events.append(t_max)

        # Fit the weibull
        wbf = WeibullFitter()

        t_failures = NotImplemented
        event_observed = t_failures != t_max

        wbf.fit(durations=t_failures, event_observed=event_observed)

        self.wbf = wbf

    def expected_condition(self):
        """Get the expected condition for a failure mode"""
        expected = dict()
        for cond_name in self.conditions.values():
            expected[cond_name] = np.array(
                [self._timelines[x][cond_name] for x in self._timelines]
            ).mean(axis=0)

        return expected

    def expected_condition_loss(self, stdev=1):
        """Get the expected condition for a failure mode"""
        expected = dict()
        for cond_name, condition in self.conditions.items():

            ec = np.array([self._timelines[x][cond_name] for x in self._timelines])

            mean = condition.perfect - ec.mean(axis=0)
            sd = ec.std(axis=0)
            upper = mean + sd * stdev
            lower = mean - sd * stdev

            upper[upper > condition.perfect] = condition.perfect
            lower[lower < condition.failed] = condition.failed

            expected[cond_name] = dict(
                lower=lower,
                mean=mean,
                upper=upper,
                sd=sd,
            )

        return expected

    def expected_risk_cost_df(self, t_start=0, t_end=None):
        """ A wrapper to turn risk cost into a df for plotting"""
        erc = self.expected_risk_cost()

        # Set the end to the largest time if no time is given
        if t_end == None:
            t_end = t_start
            for task in erc.values():
                t_end = max(max(task["time"], default=t_start), t_end)

        df = pd.DataFrame(erc).T.apply(fill_blanks, axis=1, args=(t_start, t_end))
        df.index.name = "task"
        df_cost = df.explode("cost")["cost"]
        df = df.explode("time")
        df["cost"] = df_cost

        # Add a cumulative cost
        df["cost_cumulative"] = df.groupby(by=["task"])["cost"].transform(
            pd.Series.cumsum
        )

        return df.reset_index()

    def expected_risk_cost(self, scaling=None):
        if scaling == None:
            scaling = self._sim_counter

        profile = self._expected_cost(scaling=scaling)
        profile["risk"] = self._expected_risk(scaling=scaling)

        return profile

    def _expected_cost(self, scaling=1):

        task_cost = dict()

        # Get the costs causes by tasks
        for task_name, task in self.tasks.items():
            if task.active:
                task_cost[task_name] = task.expected_costs(scaling)

        return task_cost

    def _expected_risk(self, scaling=1):
        # TODO expected risk with or without replacement

        t_failures = []
        for timeline in self._timelines.values():
            if timeline["failure"].any():
                t_failures.append(np.argmax(timeline["failure"]))

        time, cost = np.unique(t_failures, return_counts=True)
        cost = cost * self.cof.get_cost() / scaling

        return dict(time=time, cost=cost)

    def expected_tasks(self):

        task_count = dict()

        for task_name, task in self.tasks.items():
            if task.active:
                task_count[task_name] = task.expected_counts(self._sim_counter)

        return task_count

    # ****************** Reset Routines **************

    def reset_condition(self):

        # Reset conditions
        for condition in self.conditions.values():
            condition.reset()

    def reset_for_next_sim(self):

        # Reset state
        self.set_states(self.init_states)

        # Reset conditions
        for condition in self.conditions.values():
            condition.reset()

    def reset(self):

        # Reset tasks
        for task in self.tasks.values():
            task.reset()

        # Reset conditions
        for condition in self.conditions.values():
            condition.reset()

        # Reset timelines
        self.timeline = dict()
        self.timelines = dict()

        # Reset counters
        self.sim_counter = 0

    # ****************** Optimise routines ***********

    def optimal_strategy(self):
        """
        Ouptut - Run To Failure, Scheduled Replacement, Scheduled Repair, On Condition Replacement, On Condition Replacement
        """

        # For each increment

        # For each simulation

        # Record outputs

        # Cost
        # Consequence
        # Reliability
        # n_failures
        # Availability
        # Uptime
        # Downtime

        # Plot outputs

        # Return optimal strategy

    # ****************** Interface methods ***********

    def plot_timeline(self, timeline=None):
        if timeline is None:
            timeline = self.timeline

        fig, (ax_state, ax_cond, ax_task) = plt.subplots(1, 3)

        fig.set_figheight(4)
        fig.set_figwidth(24)

        ax_cond.set_title("Condition")
        ax_state.set_title("State")
        ax_task.set_title("Task")

        for condition in self.conditions:
            ax_cond.plot(timeline["time"], timeline[condition], label=condition)
            ax_cond.legend()

        for state in self.get_states():
            ax_state.plot(timeline["time"], timeline[state], label=state)
            ax_state.legend()

        for task in self.tasks:
            ax_task.plot(timeline["time"], timeline[task], label=task)
            ax_task.legend()

        plt.show()

    def update_from_dict(self, dict_data):

        for key, value in dict_data.items():

            if key in ["name", "active", "pf_curve", "pf_interval", "pf_std"]:
                self.__dict__[key] = value

            # elif key == "untreated":
            # self.set_untreated(dict_data[key])

            # elif key == "conditions":
            # self.set_conditions(dict_data[key])

            elif key == "consequence":
                self.set_consequence(dict_data[key])

            elif key == "states":
                self.set_states(dict_data[key])

            elif key == "tasks":
                self.set_tasks(dict_data[key])

            else:
                print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

    def get_value(self, key):
        """
        Takes a key as either a dictionary or a variable name and returns the value stored at that location.

        Usage:
            fm = FailureMode(name = "early_life",(alpha=10, beta = 3, gamma=1))

            #>>> dist.get_value(key="name")
            "early_life"

            #>>> dist.get_value(key={"untreated":"alpha"})
            "10

            #>>> dist.get_value(key={"untreated":("alpha", "beta")})
            (10, 3)

        """

        """
        if keys[0] in ["name", "active", "pf_curve", "pf_interval", "pf_std"]:
            self.__dict__[keys[0]] = value

        At the outside it'll have be dicts to give you the option to call multiple values long term
        At the inside its probably going to be a list

        ['untreated']['alpha']
        ['untreated']['beta']

        ['untreated']['alpha', 'beta]

        """

        """
        key_1 = #get first leve key
        key_2 =
        key_3 =
        """

        """
        for keys_1 in level_1_keys:
            # Check if it is a value
            
            # Check if it is is an object with its own get_value method

        """

        # Component
        key = {"failure_mode": {"early_life": {"tasks": {"repair": "t_interval"}}}}

        # failure mode
        key = {"untreated": "gamma"}

        if isinstance(key, str):
            value = self.__dict__[key]

        elif isinstance(key, list):
            if len(key) == 1:
                value = self.__dict__[key[0]]
            else:
                value = ()
                for k in key:
                    value = value + (self.__dict__[k],)
        else:
            print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

        return value

    def get_dash_ids(self, prefix="", sep="-"):
        """ Return a list of dash ids for values that can be changed"""

        prefix = prefix + "FailureMode" + sep + self.name + sep

        # Failure modes
        fm_ids = [
            prefix + param for param in ["active", "pf_curve", "pf_interval", "pf_std"]
        ]

        # Failure Dist
        fd_ids = self.untreated.get_dash_ids(prefix=prefix)

        # Tasks
        task_ids = []
        for task in self.tasks.values():
            task_ids = task_ids + task.get_dash_ids(prefix=prefix + "tasks" + sep)

        dash_ids = fm_ids + fd_ids + task_ids

        return dash_ids

    def get_objects(self, prefix="", sep="-"):

        # Failure mode object
        prefix = prefix + "FailureMode" + sep
        objects = [prefix + self.name]

        # Tasks objects
        prefix = prefix + self.name + sep
        objects = objects + [
            prefix + "tasks" + sep + "Task" + sep + task for task in self.tasks
        ]

        return objects

    # ****************** Demonstration ***********

    def set_demo(self):
        return self.load(demo.failure_mode_data["slow_aging"])


def doctest_setup():
    from pof.failure_mode import FailureMode
    from pof.distribution import Distribution


if __name__ == "__main__":
    import doctest

    # Add a line to set config to test version**
    doctest.testmod(
        optionflags=doctest.ELLIPSIS,
        extraglobs={"fm": FailureMode()},
    )
    print("FailureMode - Ok")
