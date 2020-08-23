"""

Author: Gavin Treseder
"""

#TODO make sure active is working

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from matplotlib import pyplot as plt
from random import random, seed

from tqdm import tqdm
from lifelines import WeibullFitter

from condition import Condition
from distribution import Distribution
from consequence import Consequence
from task import Inspection, OnConditionRepair, OnConditionReplace, ImmediateMaintenance

# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO Change this to update timeline based on states that have changed
# TODO make it work with non zero start times

seed(1)

class FailureMode:  # Maybe rename to failure mode
    def __init__(self, alpha=None, beta=None, gamma=None):

        # Failure behaviour
        self.active = True
        self.failure_dist = Distribution(alpha=alpha, beta=beta, gamma=gamma)
        self.init_dist = None

        self.pf_interval = 5  # TODO

        self.conditions = dict()

        # Failure information
        self.name = str('fm')
        self.t_fm = 0
        self.t_uptime = 0
        self.t_downtime = 0
        self.cof = Consequence()  # TODO change to a consequence model
        self.pof = None  # TODO

        # Failre Mode state
        self.states = dict()

        # Tasks
        self.task_order = [1, 2, 3, 4]  # 'inspect', 'replace', repair' # todo
        self.tasks = dict()

        # Prepare the failure mode
        self.calc_init_dist()  # TODO make this method based on a flag

        # kpis? #TODO
        # Cost and Value of current task? #TODO

        self._timelines = dict()
        self.value = None  # TODO

        self._sim_counter = 0

        return

    # ************** Set Functions *****************


    def set_failure_dist(self, failure_dist):
        self.failure_dist = failure_dist

    def set_tasks(self, tasks):
        """
        Takes a dictionary of tasks and sets the failure mode tasks
        """

        for task_name, task in tasks.items():
            self.tasks[task_name] = task

        return True

    def set_states(self, states):

        for state_name, state in states.items():
            self.states[state_name] = state

        return True

    def set_conditions(self, conditions):
        """
        Takes a dictionary of conditions and sets the failure mode conditions
        """

        for cond_name, condition in conditions.items():
            self.conditions[cond_name] = condition

        return True

    # ************** Get Functions *****************

    def get_states(self):
        return self.states

    # ************** Is Function *******************

    def is_failed(self):
        return self.states["failure"]

    def is_initiated(self):
        return self.states["initiation"]

    def is_detected(self):
        return self.states["detection"]

    # ******

    def calc_init_dist(self):  # TODO needs to get passed a condition and a pof
        """
        Convert the probability of failure into a probability of initiation
        """

        # Super simple placeholder # TODO add other methods
        alpha = self.failure_dist.alpha
        beta = self.failure_dist.beta
        gamma = self.failure_dist.gamma - self.pf_interval

        self.init_dist = Distribution(alpha=alpha, beta=beta, gamma=gamma)

        return

    def get_expected_condition(self, t_min, t_max):  # TODO retire?

        t_forecast = np.linspace(t_min, t_max, t_max - t_min + 1, dtype=int)

        # Calculate the probability of initiation for the time period
        prob_initiation = f_ti[t_forecast[1:]]

        # Add the probability after t_max onto the final row with perfect condition
        prob_not_considered = sf_i[t_max]
        prob_initiation = np.append(prob_initiation, prob_not_considered)

        # Scale to ignore the probability that occurs before t_min
        prob_initiation = prob_initiation / prob_initiation.sum()

        # Create a condition matrix of future condition
        deg_matrix = np.tril(
            np.full((deg_curve.size, deg_curve.size), 100), -1
        ) + np.triu(circulant(deg_curve).T)

        condition_outcomes = (deg_matrix.T * prob_initiation).T

        condition_mean = condition_outcomes.sum(axis=0)

        return True

        # Need to consider condition that has a probability of outcomes ()

    # *************** Condition Loss *************** # TODO not updated to new method

    def measure_condition_loss(self):

        return self.condition.measure()

    def get_cond_loss(self):

        # New Condition at end of t

        return

    def get_p_initiation(
        self, t_step
    ):  # TODO make a robust time step. (t_min, t_max, etc)

        if self.is_initiated() == True:
            p_i = 1
        else:
            p_i = self.init_dist.conditional_f(
                self.t_fm, self.t_fm + t_step
            )  # TODO update to actual time

        return p_i

    def sim_event(self):

        # Find first task to be actioned

        # Complete task triggers

        # Refresh the timeline where maintenance or intervention

        # Increment maint

        return NotImplemented

    # ****************** Timeline ******************

    def mc_timeline(self, t_end, t_start=0, n_iterations=100):

        self.reset()  # TODO ditch this

        for i in tqdm(range(n_iterations)):
            self._timelines[i] = self.sim_timeline(t_end=t_end, t_start=t_start)

    def sim_timeline(self, t_end, t_start=0, verbose=False):

        timeline = self.init_timeline(t_start=t_start, t_end=t_end)

        t_now = t_start

        while t_now < t_end:

            # Check when the next task needs to be executed
            t_now, task_names = self.next_tasks(timeline, t_now, t_end)

            # Complete those tasks
            self.complete_tasks(t_now, task_names, verbose=verbose)

            t_now = t_now + 1

        self._sim_counter = self._sim_counter + 1
        self.reset_for_next_sim()

        return self.timeline

    def complete_tasks(self, t_now, task_names, verbose=False):
        """ Executes the tasks """
        for task_name in task_names:
            if verbose:
                print(t_now, task_names)

            # Complete the tasks
            states = self.tasks[task_name].sim_completion(
                t_now,
                timeline=self.timeline,
                states=self.get_states(),
                conditions=self.conditions,
                verbose=verbose,
            )

            # Update timeline
            
            self.set_states(states)  
            self.update_timeline(t_now + 1, updates=states, verbose=verbose)

    def init_timeline(self, t_end, t_start=0):
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
            t_initiate = min(t_end, int(self.init_dist.sample()))
            timeline["initiation"][t_initiate:] = 1

        # Get condtiion
        for condition_name, condition in self.conditions.items():
            timeline[condition_name] = condition.get_condition_profile(
                t_start=t_start - t_initiate, t_stop=t_end - t_initiate
            )

        # Check failure
        timeline["failure"] = np.full(t_end + 1, self.is_failed())
        if not self.is_failed():
            for condition in self.conditions.values():
                tl_f = condition.sim_failure_timeline(
                    t_start=t_start - t_initiate, t_stop=t_end - t_initiate
                )
                timeline["failure"] = (timeline["failure"]) | (tl_f)

        # Check tasks with time based trigger
        for task in self.tasks.values():

            if task.trigger == "time":
                timeline[task.activity] = task.sim_timeline(t_end)

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
            t_end = self.timeline['time'][-1]

        # Initiation -> Condition -> time_tasks -> states -> tasks
        if t_start < t_end:

            if "time" in updates:
                self.timeline["time"] = np.linspace(
                    t_start, t_end, t_end - t_start + 1, dtype=int
                )

            # Check for initiation changes
            if "initiation" in updates:
                t_initiate = min(
                    t_end, t_start + int(self.init_dist.sample())
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
                    self.timeline[condition_name][
                        t_start:
                    ] = condition.get_condition_profile(
                        t_start=t_start - t_initiate, t_stop=t_end - t_initiate
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

            # Check tasks with time based trigger
            for task_name, task in self.tasks.items():

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

    def next_tasks(self, timeline=None, t_start=0, t_end=None):
        """
        Takes a timeline and returns the next time, and next task that will be completed
        """

        if timeline is None:
            timeline=self.timeline

        if t_end is None:
            t_end = timeline["time"][-1]

        next_tasks = []
        next_time = t_end

        # TODO make this more efficient by using a task array rather than a for loop
        for task in self.tasks: 

            if 0 in timeline[task][t_start:]:
                t_task = (
                    timeline["time"][np.argmax(timeline[task][t_start:] == 0)] + t_start
                )

                if t_task < next_time:
                    next_tasks = [task]
                    next_time = t_task
                elif t_task == next_time:
                    next_tasks = np.append(next_tasks, task)

        return next_time, next_tasks


    # ****************** Expected Methods  ************

    def expected(self):
        """Returns all expected outcomes"""
        NotImplemented

    def expected_pof(self):
        
        
        NotImplemented

    def expected_pof_c(self):
        """ Return the probability """

        t_failures = []
        for timeline in self._timelines.values():
            t_failures = np.append(t_failures, np.argmax(timeline["failure"]))

        # Arange into failures and censored data
        failures = t_failures[t_failures > 0]
        censored = np.full(sum(t_failures == 0), 200)

        return failures


    def expected_cost_dict(self):  # TODO legacy

        d_tc = dict()

        for task_name, task in self.tasks.items():

            time, quantity = np.unique(task.t_completion, return_counts=True)
            cost = quantity * task.cost

            d_tc[task_name] = dict(time=time, cost=cost,)

        return d_tc

    def expected_costs(self):

        cost = dict()

        for task_name, task in self.tasks.items():
            cost[task_name] = len(task.t_completion) * task.cost / self._sim_counter

        t_failures = []
        for timeline in self._timelines.values():
            t_failures = np.append(t_failures, np.argmax(timeline["failure"]))

        # Arange into failures and censored data
        failures = t_failures[t_failures > 0]

        cost["risk"] = len(failures) * self.cof.risk_cost_total / self._sim_counter

        return cost

    def mc_failures_df(self):

        t_failures = []
        for timeline in self._timelines.values():
            t_failures = np.append(t_failures, np.argmax(timeline["failure"]))

        # Arange into failures and censored data
        failures = t_failures[t_failures > 0]
        censored = np.full(sum(t_failures == 0), 200)

        return failures

    def mc_risk_df(self):

        t_failures = []
        for timeline in self._timelines.values():
            t_failures = np.append(
                t_failures, timeline["time"][np.argmax(timeline["failure"])]
            )

        # Arange into failures and censored data
        failures = t_failures[t_failures > 0]

        time, quantity = np.unique(failures, return_counts=True)
        cost = quantity * self.cof.risk_cost_total / self._sim_counter
        cost_cumulative = cost.cumsum()

        df = pd.DataFrame(
            dict(
                cost_type="risk",
                time=time,
                quantity=quantity,
                cost=cost,
                cost_cumulative=cost_cumulative,
            )
        )

        new_index = pd.Index(np.arange(0, 200, 1), name="time")

        df = df.set_index("time").reindex(new_index).reset_index()
        df.loc[0, :] = 0
        df.loc[0, "task"] = "risk"
        df = df.fillna(method="ffill")

        return df

    def expected_cost_df(self):

        df_tasks = pd.DataFrame()
        new_index = pd.Index(np.arange(0, 200, 1), name="time")

        for task_name, task in self.tasks.items():

            time, quantity = np.unique(task.t_completion, return_counts=True)
            cost = quantity * task.cost / self._sim_counter
            cost_cumulative = cost.cumsum()

            df = pd.DataFrame(
                dict(
                    task=task_name,
                    time=time,
                    cost=cost,
                    cost_cumulative=cost_cumulative,
                )
            )

            df = df.set_index("time").reindex(new_index).reset_index()
            df.loc[0, :] = 0
            df.loc[0, "task"] = task_name
            df = df.fillna(method="ffill")

            df_tasks = pd.concat([df_tasks, df])

        df_tasks = pd.concat((df_tasks, self.mc_risk_df()))

        return df_tasks

    # ****************** Reset Routines **************

    def reset_for_next_sim(self):

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

    def dash_update(self, dash_id, value, sep='-'):
        """Updates a the failure mode object using the dash componenet ID"""

        try:
            
            next_id = dash_id.split(sep)[0]
        
            # Check if the next component is a param of 
            if next_id in ['active']:

                self.active = value

            elif next_id == 'failure_dist':

                dash_id = dash_id.replace(next_id + sep, "")
                self.failure_dist.dash_update(dash_id, value)

            elif next_id == 'task':
      
                dash_id = dash_id.replace(next_id + sep, "")
                task_name= dash_id.split(sep)[0]
                dash_id = dash_id.replace(task_name + sep, "")
                self.tasks[task_name].dash_update(dash_id, value)

        except:

            print("Invalid dash component %s" %(dash_id))


    def get_dash_ids(self, prefix="", sep='-'):
        """ Return a list of dash ids for values that can be changed"""

        # Failure modes
        fm_ids = [prefix + param for param in ['active']]

        # Failure Dist
        fd_prefix = prefix + 'failure_dist' + sep
        fd_ids = self.failure_dist.get_dash_id(prefix=fd_prefix)

        # Tasks
        task_ids = []
        for task_name, task in self.tasks.items():
            task_prefix = prefix + 'task' + sep + task_name + sep
            task_ids = task_ids + task.get_dash_ids(prefix=task_prefix)

        dash_ids = fm_ids + fd_ids + task_ids

        return dash_ids

    # ****************** Demonstration ***********

    def reset_demo(self):
        self.failure_dist=None
        self.conditions=None
        self.tasks = None
        self.states = None
        self.set_demo()

    def set_demo(self):

        if self.failure_dist is None:
            self.set_failure_dist(Distribution(alpha=50, beta=3, gamma=10))

        if not self.conditions:
            self.set_conditions(
                dict(
                    wall_thickness=Condition(100, 0, "linear", [-5]),
                    external_diameter=Condition(100, 0, "linear", [-2]),
                )
            )

        if not self.tasks:
            self.set_tasks(
                dict(
                    inspection=Inspection(t_interval=5, t_delay=10).set_default(),
                    on_condition_repair=OnConditionRepair(activity="on_condition_repair").set_default(),
                    cm=ImmediateMaintenance(activity="cm").set_default(),
                )
            )

        if not self.states:
            self.set_states(dict(initiation=False, detection=False, failure=False,))

        # Prepare the failure mode
        self.calc_init_dist()

        return self

if __name__ == "__main__":
    failure_mode = FailureMode()
    print("FailureMode - Ok")
