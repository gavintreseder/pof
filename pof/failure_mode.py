"""

Author: Gavin Treseder
"""


# Change the system path is
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
import collections
from scipy.linalg import circulant
from matplotlib import pyplot as plt
from random import random, seed
from enum import Enum

from tqdm import tqdm
from lifelines import WeibullFitter

from pof.helper import fill_blanks, id_update
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
from pof.config import FailureModeConfig as cf


# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO Change this to update timeline based on states that have changed
# TODO make it work with non zero start times

seed(1)


class FailureMode:  # Maybe rename to failure mode

    PF_CURVE = ["linear", "step"]

    def __init__(
        self,
        name="fm",
        active=True,
        pf_curve="linear",
        pf_interval=10,
        pf_std=0,
        untreated=dict(),
        conditions=None,  # TODO this needs to change for condition loss
        consequence=dict(),
        states=dict(),
        tasks=dict(),
        *args,
        **kwargs,
    ):

        # Failure Information
        self.name = name
        self.active = active
        self.pf_curve = pf_curve
        self.pf_interval = pf_interval
        self.pf_std = pf_std

        # Failure Distributions
        self.untreated = None
        self.init_dist = None
        self.pof = None
        self.set_untreated(untreated)

        # Indicators

        # Faiure Condition
        self.conditions = dict()
        self.set_conditions(conditions)

        # Consequence of failure
        self.cof = None
        self.set_consequence(consequence)

        # Failure Mode state
        self.states = dict()
        self.set_states(states)

        # Init State
        self.init_states = dict()  # TODO change to asset info set
        self.set_init_state(states)

        # Tasks
        self.tasks = dict()
        self.set_tasks(tasks)

        # Simulation details
        self.timeline = dict()
        self._timelines = dict()
        self._sim_counter = 0

    # ************** Load Functions *****************

    @classmethod
    def load(cls, details=None):
        try:
            if isinstance(details, dict):
                fm = cls.from_dict(details)
            else:
                fm = cls()
        except:
            raise ValueError("Error loading %s data" % (cls.__name__))
        return fm

    @classmethod
    def from_dict(cls, details=None):
        try:
            fm = cls(**details)
        except:
            raise ValueError("Error loading %s data from dictionary" % (cls.__name__))
        return fm

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

                    self.conditions[cond_name] = ConditionIndicator.from_dict(
                        condition
                    )  # Gav's fix
                    # TODO Illyse - this doesn't work for all methods
                    """
                    if self.conditions is not None: 
                        for key, value in condition.items():
                            self.conditions[key] = value

                    else:
                        self.conditions[cond_name] = ConditionIndicator.from_dict(
                            condition
                        )
                    """
                else:
                    print(
                        'ERROR: Cannot update "%s" condition from dict'
                        % (self.__class__.__name__)
                    )

    def set_untreated(self, untreated):

        # Load a distribution object
        if isinstance(untreated, Distribution):
            self.untreated = untreated

        # Add a name to the distribution and set create the object
        elif isinstance(untreated, dict):
            if self.untreated is not None:
                for key, value in untreated.items():
                    self.untreated[key] = value

            else:
                untreated["name"] = "untreated"
                self.untreated = Distribution.from_dict(untreated)

        else:
            print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

        # Set the probability of initiation using the untreated parameters
        self.set_init_dist()

    def set_init_state(self, states):
        # TODO fix this default
        for state in ["detection", "failure", "initiation"]:
            self.init_states[state] = False

        for state_name, state in states.items():
            self.init_states[state_name] = bool(state)

        return True

    def set_states(self, states):
        # TODO check this on the wekeend and split into set and update methods. Set at start. Update
        for state_name, state in states.items():
            self.states[state_name] = bool(state)

        for state in ["detection", "failure", "initiation"]:
            if state not in self.states:
                self.states[state] = False

    def set_tasks(self, tasks):
        """
        Takes a dictionary of tasks and sets the failure mode tasks
        """

        for task_name, task in tasks.items():
            if isinstance(task, Task):
                self.tasks[task_name] = task
            elif isinstance(task, dict):
                if task["activity"] == "Inspection":
                    self.tasks[task["name"]] = Inspection().load(task)
                elif task["activity"] == "ConditionTask":
                    self.tasks[task_name] = ConditionTask().load(task)

                else:
                    print("Invalid Task Activity")
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
        if isinstance(indicator, collections.Iterable):
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

        # Super simple placeholder # TODO add other methods
        alpha = self.untreated.alpha
        beta = self.untreated.beta
        if self.pf_interval is None:
            gamma = max(0, self.untreated.gamma)
        else:
            gamma = max(0, self.untreated.gamma - self.pf_interval)

        # TODO add an adjustment to make sure the pfinterval results in a resaonable gamma
        # self.pf_interval = self.pf_interval - max(self.gamma - self.pf_interval + self.pf_std * 3)

        self.init_dist = Distribution(alpha=alpha, beta=beta, gamma=gamma, name="init")

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

    def _expected_risk(
        self, scaling=1
    ):  # TODO expected risk with or without replacement

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
        self._timelines = dict()
        self.timeline = dict()

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

    def update2(self, id_object, value=None):
        """"""
        if isinstance(id_object, str):
            self.update_from_str(id_object, value, sep="-")

        elif isinstance(id_object, dict):
            self.update_from_dict(id_object)

        else:
            print(
                'ERROR: Cannot update "%s" from string or dict'
                % (self.__class__.__name__)
            )

    def update_from_str(self, id_str, value, sep="-"):

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

        if keys[0] in ["name", "active", "pf_curve", "pf_interval", "pf_std"]:
            self.__dict__[keys[0]] = value

        elif keys[0] == "untreated":
            self.set_untreated(dict_data[keys[0]])

        elif keys[0] == "condition":
            self.set_conditions(dict_data[keys[0]])

        elif keys[0] == "consequence":
            self.set_consequence(dict_data[keys[0]])

        elif keys[0] == "state":
            self.set_states(dict_data[keys[0]])

        elif keys[0] == "task":
            self.set_tasks(dict_data[keys[0]])

        else:
            print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

    def update(self, id_str, value, sep="-"):
        """Updates a the failure mode object using the dash componenet ID"""

        try:

            # Remove the class type and class name from the dash_id
            id_str = id_str.split(self.name + sep, 1)[1]
            var = id_str.split(sep)[0]

            # Check if the variable is an attribute of the class
            if var in self.__dict__:

                # Check if the variable is a dictionary
                if isinstance(self.__dict__[var], dict):

                    var_2 = id_str.split(sep)[1]

                    # Check if the variable is a class with its own update methods
                    if var_2 in ["Condition", "Task", "Distribution"]:
                        var_3 = id_str.split(sep)[2]
                        self.__dict__[var][var_3].update(id_str, value, sep)

                        self.set_init_dist()  # TODO Ghetto fix
                        for condition in self.conditions.values():  # TODO Ghetto fix
                            condition.pf_interval = value  # TODO Ghetto fix
                    else:
                        self.__dict__[var][var_2] = value
                else:
                    self.__dict__[var] = value
                    if var == "pf_interval":  # TODO Ghetto fix
                        self.set_init_dist()  # TODO Ghetto fix
                        for condition in self.conditions.values():  # TODO Ghetto fix
                            condition.pf_interval = value  # TODO Ghetto fix

            # Check if the variable is a class instance
            else:

                var = id_str.split(sep)[1]

                if var in self.__dict__ and isinstance(
                    self.__dict__[var], (Indicator, Distribution, Task)
                ):
                    self.__dict__[var].update(id_str, value, sep)
                    if var == "untreated":
                        self.set_init_dist()  # TODO Ghetto fix
                else:
                    print('Invalid id "%s" %s not in class' % (id_str, var))

        except:
            print("Invalid ID")

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


if __name__ == "__main__":

    failure_mode = FailureMode()
    print("FailureMode - Ok")
