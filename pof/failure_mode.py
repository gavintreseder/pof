"""

Author: Gavin Treseder
"""

# ************ Packages ********************

import copy
from dataclasses import dataclass, field
import logging
from typing import Dict, Optional
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

from config import config
from pof.helper import fill_blanks, id_update, str_to_dict
from pof.indicator import Indicator, ConditionIndicator
from pof.distribution import Distribution, DistributionManager
from pof.consequence import Consequence
from pof.task import (
    Task,
    Inspection,
    ConditionTask,
)
import pof.demo as demo
from pof.load import Load


# TODO Use condition pf to change indicator
# #TODO task indicator

# TODO Change this to update timeline based on states that have changed
# TODO make it work with non zero start times

cf = config["FailureMode"]

seed(1)


class FailureModeData(Load):
    """
    A class that contains the data for the FailureMode object.

    This is a temporary fix due to issues with @property and @dataclass changing the constructor to only accept data with leading underscores
    """

    # pylint: disable=too-many-instance-attributes
    # Reasonable in this implementation

    name: str = "fm"
    active: bool = True
    pf_curve: str = "step"
    pf_interval: int = 0
    pf_std: int = 0

    # Set methods used
    dists: Dict = None
    consequence: Dict = None
    indicators: Dict = None
    conditions: Dict = None
    states: Dict = None
    tasks: Dict = None
    init_states: Dict = None

    # Simulation Details
    timeline: Dict = field(init=False, repr=False, default_factory=lambda: dict())
    timelines: Dict = field(init=False, repr=False, default_factory=lambda: dict())
    sim_counter: int = 0

    untreated: Distribution = None


class FailureMode(Load):

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

    """

    # TODO check why failure_mode is included

    # *************** Property Methods *************
    # #TODO convert any get/set pairs to properties

    # Class Variables
    PF_CURVES = ["linear", "step"]
    REQUIRED_STATES = ["initiation", "detection", "failure"]

    def __init__(
        self,
        name: str = "fm",
        active: bool = True,
        pf_curve: str = "step",
        pf_interval: int = 0,
        pf_std: int = 0,
        untreated=None,
        consequence: Dict = None,
        indicators: Dict = None,
        conditions: Dict = None,
        states: Dict = None,
        tasks: Dict = None,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.dists = dict()
        self.indicators = dict()
        self.conditions = dict()
        self.consequences = dict()
        self.tasks = dict()
        self.init_states = dict()
        self.states = dict()

        self.active = active
        self.pf_curve = pf_curve
        self.pf_interval = pf_interval
        self.pf_std = pf_std
        self.conditions_to_update = set()  # not used yet

        self.untreated = untreated
        self.set_indicators(indicators)
        self.set_conditions(conditions)
        self.set_consequence(consequence)
        self.set_tasks(tasks)
        self.set_init_states(states)
        self.set_states(states)

        self.timeline = dict()
        self._timelines = dict()
        self._sim_counter = 0

    @property
    def pf_curve(self):
        return self._pf_curve

    @pf_curve.setter
    def pf_curve(self, value):
        """ Set the pf_curve to a valid str"""

        if value in self.PF_CURVES:
            self._pf_curve = value
        else:
            raise ValueError(
                "%s (%s) - pf_curve must be from: %s"
                % (self.__class__.__name__, self.name, self.PF_CURVES)
            )

    @property
    def pf_interval(self):
        return self._pf_interval

    @pf_interval.setter
    def pf_interval(self, value):

        try:
            if value >= 0:
                self._pf_interval = value
                if "untreated" in self.dists:
                    self._set_init()
            else:
                raise ValueError(
                    "%s (%s) - pf_interval must be greater than 0"
                    % (self.__class__.__name__, self.name)
                )
        except:
            raise

    @property
    def untreated(self):
        return self.dists.get("untreated", None)

    @untreated.setter
    def untreated(self, dist):
        if dist is not None:
            if isinstance(dist, Distribution):
                dist.name = "untreated"
            else:
                dist["name"] = "untreated"

            self.set_dists({"untreated": dist})
            self._set_init()

    def _set_init(self):
        init = Distribution.from_pf_interval(self.dists["untreated"], self.pf_interval)
        init.name = "init"
        self.set_dists({"init": init})

    # ************** Set Functions *****************

    def set_consequence(self, consequence=NotImplemented):

        self.consequence = Consequence()

    def set_indicators(self, var=None):

        self.set_obj("indicators", Indicator, var)

    def set_conditions(self, var=None):
        """
        Takes a dict of conditions and saves them so they can be used to call indicators
        """
        # TODO Add checks to create the condition if it doesn't exist
        # TODO Make this work for different pf_intervals for different conditions

        if bool(var):
            if "name" in var:
                self.conditions = {var["name"]: var}
            else:
                self.conditions = var
            # Create an indicator for any conditions not in the indicator list

            for cond_name, condition in self.conditions.items():
                if cond_name not in self.indicators:
                    indicator = ConditionIndicator.load(condition)
                    self.set_indicators(indicator)
        else:
            # Create a simple indicator
            # Removing this temporarilty #TODO
            indicator = ConditionIndicator(
                name=self.name,
                pf_curve="step",
                pf_interval=0,
                pf_std=0,
                perfect=False,
                failed=True,
            )
            self.conditions = {self.name: {}}
            self.set_indicators(indicator)

    def set_dists(self, dists):
        self.set_obj("dists", Distribution, dists)

    def set_init_states(self, states):
        # TODO Update this method at the same time as set state

        if self.init_states is None:
            self.init_states = {state: False for state in self.REQUIRED_STATES}

        # update from the input argument
        if states is not None:
            self.init_states.update(states)

        # Update from required states
        for state in self.REQUIRED_STATES:
            if state not in self.init_states:
                self.init_states[state] = False

    def set_states(self, states=None):
        # TODO check this on the wekeend and split into set and update methods. Set at start. Update
        # TODO make this work with actual asset
        # TODO this is super super ugly. make state an indiciator or it's own class.

        # Set a default value if none has been provided
        if self.states is None:
            if cf.get("use_default", config.get("Load").get("use_default")):
                self.states = {state: False for state in self.REQUIRED_STATES}
            else:
                raise ValueError("Failure Mode - %s - No states provided" % (self.name))

        # update from the input argument
        if states is not None:
            self.states.update(states)

        # Update from required states
        for state in self.REQUIRED_STATES:
            if state not in self.states:
                self.states[state] = False

    def set_tasks(self, tasks):
        """
        Takes a dictionary of tasks and sets the failure mode tasks
        """
        self.set_obj("tasks", Task, tasks)

    def link_indicator(self, indicator):
        """
        Takes an indicator ojbect, or an iterable list of objects and links condition
        """

        # TODO dlete

        if indicator is None:
            self.indictors = dict()
        else:
            # If it is an iterable, link them all
            if isinstance(indicator, Iterable):
                for indicator in indicator.values():
                    if indicator.name in self.conditions:
                        self.indicators[indicator.name] = indicator

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

    def get_pf_interval(self, cond_name=None):
        pf_interval = self.conditions.get(cond_name, {}).get(
            "pf_interval", self.pf_interval
        )
        if pf_interval is None:
            pf_interval = self.pf_interval
        return pf_interval

    def get_pf_std(self, cond_name=None):
        return self.conditions.get(cond_name, {}).get("pf_std", self.pf_std)

    # ************** Is Function *******************

    def is_failed(self):
        return self.states["failure"]

    def is_initiated(self):
        return self.states["initiation"]

    def is_detected(self):
        return self.states["detection"]

    # *************** Get methods *******************

    def get_expected_pof(self):

        # TODO add a check if it has been simulated yet self.pof is None, self._timlines = None

        return self.pof

    # ****************** Timeline ******************

    def mc_timeline(self, t_end, t_start=0, n_iterations=100):

        self.reset()  # TODO ditch this

        for i in tqdm(range(n_iterations)):
            self.sim_timeline(t_end=t_end + i, t_start=t_start)
            self.increment_counter()
            self.save_timeline(i)
            self.reset_for_next_sim()

    def sim_timeline(self, t_end, t_start=0):

        timeline = self.init_timeline(t_start=t_start, t_end=t_end)

        if self.active:

            t_now = t_start

            while t_now < t_end:

                # Check when the next task needs to be executed
                t_now, task_names = self.next_tasks(timeline, t_now, t_end)

                # Complete those tasks
                system_impact = self.complete_tasks(t_now, task_names)

                t_now = t_now + 1

                if "component" in system_impact:
                    self.renew(t_now)

        return self.timeline

    def complete_tasks(self, t_now, task_names):
        """ Executes the tasks and returns the system impact """

        system_impacts = []
        if self.active:
            for task_name in task_names:
                logging.debug("Time %s - Tasks %s", t_now, task_names)

                # Complete the tasks
                states = self.tasks[task_name].sim_completion(
                    t_now,
                    timeline=self.timeline,
                    states=self.get_states(),
                    conditions=self.indicators,
                )

                # Update timeline
                self.set_states(states)
                self.update_timeline(t_now + 1, updates=states)

                # Check if a system impact is triggered
                system_impacts = system_impacts + self.tasks[task_name].system_impact()

        return system_impacts

    def init_timeline(self, t_end, t_start=0):

        self._init_timeline(t_end, t_start)

        if self.active:

            updates = dict(
                initiation=self.is_initiated(),
                detection=self.is_detected(),
                failure=self.is_failed(),
            )
            for task in self.tasks.values():
                updates[task.name] = None

            self.update_timeline(t_start=t_start, t_end=t_end, updates=updates)

        return self.timeline

    def _init_timeline(self, t_end, t_start=0):
        """
        Simulates a single timeline to determine the state, condition and tasks
        """

        increments = t_end - t_start + 1

        timeline = dict(
            time=np.linspace(t_start, t_end, increments, dtype=int),
            initiation=np.full(increments, False),
            detection=np.full(increments, False),
            failure=np.full(increments, False),
        )

        # Update conditions for failure_mode and any conditions that trigger tasks
        for cond_name in self._cond_to_update():
            timeline[cond_name] = np.full(increments, -1)

        # Tasks
        for task in self.tasks.values():
            timeline[task.name] = np.full(increments, -1)

        self.timeline = timeline

    def _cond_to_update(self):
        cond_to_update = set(self.conditions)
        for task in self.tasks.values():
            cond_to_update = cond_to_update | set(task.get_triggers("condition"))
        return cond_to_update

    def update_timeline(self, t_start, t_end=None, updates=dict()):
        """
        Takes a timeline and updates tasks that are impacted
        """

        if t_end is None:
            t_end = self.timeline["time"][-1]

        # Initiation -> Condition -> time_tasks -> detection -> tasks
        if t_start < t_end:

            if "time" in updates:
                self.timeline["time"] = np.linspace(
                    t_start, t_end, t_end - t_start + 1, dtype=int
                )

            # Check for initiation changes
            if "initiation" in updates:
                if updates["initiation"]:
                    t_initiate = t_start
                else:
                    # TODO make this conditionalsf
                    t_initiate = min(
                        t_end + 1,
                        t_start + int(self.dists["init"].sample()),
                    )
                self.timeline["initiation"][t_start:t_initiate] = updates["initiation"]
                self.timeline["initiation"][t_initiate:] = True
            else:
                if self.timeline["initiation"][t_start:].any():
                    t_initiate = np.argmax(self.timeline["initiation"][t_start:] > 0)
                else:
                    t_initiate = t_end

            # Check for condition changes
            for cond_name in self._cond_to_update():
                if "initiation" in updates or cond_name in updates:
                    logging.debug(
                        "condition %s, start %s, initiate %s, end %s",
                        cond_name,
                        t_start,
                        t_initiate,
                        t_end,
                    )
                    # self.conditions[condition_name].set_condition(self.timeline[condition_name][t_start])
                    # #TODO this should be set earlier using a a better method
                    self.timeline[cond_name][t_start:] = self.indicators[
                        cond_name
                    ].sim_timeline(
                        t_delay=t_start,
                        t_start=t_start - t_initiate,
                        t_stop=t_end - t_initiate,
                        pf_interval=self.get_pf_interval(cond_name),
                        pf_std=self.get_pf_std(cond_name),
                    )

            # Check for failure changes
            if any(
                state in updates for state in ["initiation", "detection", "failure"]
            ):
                self.timeline["failure"][t_start:] = updates.get("failure", False)
                for cond_name in self._cond_to_update():
                    tl_f = self.indicators[cond_name].sim_failure_timeline(
                        t_delay=t_start,
                        t_start=t_start - t_initiate,
                        t_stop=t_end - t_initiate,
                    )
                    self.timeline["failure"][t_start:] = (
                        self.timeline["failure"][t_start:]
                    ) | (tl_f)

            # Update time based tasks
            for task_name, task in self.tasks.items():

                if task.trigger == "time" and task_name in updates:
                    self.timeline[task_name][t_start:] = task.sim_timeline(
                        t_start=t_start, t_end=t_end, timeline=self.timeline
                    )

            # Check for detection changes
            if "detection" in updates:
                self.timeline["detection"][t_start:] = updates["detection"]

            # Update condition based tasks if the failure mode initiation has changed
            for task_name, task in self.tasks.items():

                if task.trigger == "condition":
                    self.timeline[task_name][t_start:] = task.sim_timeline(
                        t_start=t_start,
                        t_end=t_end,
                        timeline=self.timeline,
                        indicators=self.indicators,
                    )

        return self.timeline

    def renew(self, t_renew):
        """
        Update timeline
        """
        if cf.get("remain_failed"):
            self.fail(t_renew)
        else:
            self.replace(t_renew)

    def fail(self, t_fail):
        """ Cut the timeline short and prevent any more tasks from triggering"""

        # Make timelines shorter
        for state in ["time", "initiation", "detection", "failure"]:
            self.timeline[state] = self.timeline[state][:t_fail]

        # Cancel future tasks
        for task in self.tasks.values():
            # TODO consider moving timeline to the task and make sure task timeline canges to zero
            self.timeline[task.name][t_fail:] = -1

    def replace(self, t_replace):
        """ Update the asset to a perfect asset """
        state_after_replace = dict(initation=False, detection=False, failure=False)
        self.update_timeline(t_start=t_replace, updates=state_after_replace)

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

        for ind in self.indicators.values():
            ind.save_timeline(i)

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
        for cond_name in self.conditions:
            expected[cond_name] = np.array(
                [self._timelines[x][cond_name] for x in self._timelines]
            ).mean(axis=0)

        return expected

    def expected_condition_loss(self, stdev=1):
        """Get the expected condition for a failure mode"""
        expected = dict()
        for ind_name, indicator in self.indicators.items():

            ec = np.array([self._timelines[x][ind_name] for x in self._timelines])

            mean = indicator.perfect - ec.mean(axis=0)
            sd = ec.std(axis=0)
            upper = mean + sd * stdev
            lower = mean - sd * stdev

            upper[upper > indicator.perfect] = indicator.perfect
            lower[lower < indicator.failed] = indicator.failed

            expected[ind_name] = dict(
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
        if scaling is None:
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
        cost = cost * self.consequence.get_cost() / scaling

        return dict(time=time, cost=cost)

    def _expected_life(self):
        """ Get the expected life from each timeline"""

        expected_life = []
        for timeline in self._timelines.items():
            expected_life.append(timeline["time"][-1])

        return expected_life

    def expected_tasks(self):

        task_count = dict()

        for task_name, task in self.tasks.items():
            if task.active:
                task_count[task_name] = task.expected_counts(self._sim_counter)

        return task_count

    # ****************** Reset Routines **************

    def reset_condition(self):

        # Reset conditions
        for indicator in self.indicators.values():
            indicator.reset()

    def reset_for_next_sim(self):

        # Reset state
        self.set_states(self.init_states)

        # Reset indicators
        for ind in self.indicators.values():
            ind.reset_for_next_sim()  # TODO will this reset for all, or just for None

    def reset(self):

        # Reset tasks
        for task in self.tasks.values():
            task.reset()

        # Reset conditions
        for indicator in self.indicators.values():
            indicator.reset()  # TODO will this reset for all, or just for None

        # Reset timelines
        self.timeline = dict()
        self._timelines = dict()

        # Reset counters
        self._sim_counter = 0

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

        for cond_name in self.conditions:
            ax_cond.plot(timeline["time"], timeline[cond_name], label=cond_name)
            ax_cond.legend()

        for state in self.get_states():
            ax_state.plot(timeline["time"], timeline[state], label=state)
            ax_state.legend()

        for task in self.tasks:
            ax_task.plot(timeline["time"], timeline[task], label=task)
            ax_task.legend()

        plt.show()

    def update_from_dict(self, data):
        """Simple fix"""

        untreated = copy.copy(self.dists.get("untreated", None))

        super().update_from_dict(data)

        if untreated != self.untreated:
            self._set_init()

    # def update_from_dict(self, dict_data):

    #     for key, value in dict_data.items():

    #         if key in [
    #             "name",
    #             "active",
    #             "pf_curve",
    #             "pf_interval",
    #             "pf_std",
    #         ]:
    #             setattr(self, key, value)

    #         elif key in ["untreated", "dists"]:
    #             self.set_dists(dict_data[key])

    #         elif key == "conditions":
    #             self.set_conditions(dict_data[key])

    #         elif key == "consequence":
    #             self.set_consequence(dict_data[key])

    #         elif key == "states":
    #             self.set_states(dict_data[key])

    #         elif key == "tasks":
    #             self.set_tasks(dict_data[key])

    #         else:
    #             print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

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

        prefix = prefix + self.name + sep

        # Failure modes
        fm_ids = [
            prefix + param for param in ["active", "pf_curve", "pf_interval", "pf_std"]
        ]

        # Failure Dist
        fd_ids = self.untreated.get_dash_ids(prefix=prefix + "dists" + sep)

        # Tasks
        task_ids = []
        for task in self.tasks.values():
            task_ids = task_ids + task.get_dash_ids(prefix=prefix + "tasks" + sep)

        dash_ids = fm_ids + fd_ids + task_ids

        return dash_ids

    def get_objects(self, prefix="", sep="-"):

        # Failure mode object
        objects = [prefix + self.name]

        # Tasks objects
        prefix = prefix + self.name + sep
        objects = objects + [prefix + "tasks" + sep + task for task in self.tasks]

        return objects

    # ****************** Demonstration ***********
    @classmethod
    def demo(self):
        return self.load(demo.failure_mode_data["slow_aging"])


if __name__ == "__main__":
    import doctest

    # Add a line to set config to test version**
    doctest.testmod(
        optionflags=doctest.ELLIPSIS,
        extraglobs={"fm": FailureMode()},
    )
    print("FailureMode - Ok")
