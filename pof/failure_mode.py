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

if __package__ is None or __package__ == '':
    from helper import fill_blanks, id_update
    from condition import Condition
    from distribution import Distribution
    from consequence import Consequence
    from task import Task, Inspection, OnConditionRepair, OnConditionReplacement, ImmediateMaintenance
else:
    from pof.helper import fill_blanks, id_update
    from pof.condition import Condition
    from pof.distribution import Distribution
    from pof.consequence import Consequence
    from pof.task import Task, Inspection, OnConditionRepair, OnConditionReplacement, ImmediateMaintenance


# TODO move t somewhere else
# TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
# TODO Change this to update timeline based on states that have changed
# TODO make it work with non zero start times

seed(1)

class FailureMode:  # Maybe rename to failure mode


    PF_CURVE = ['linear', 'pf_linear'] #TODO add the rest


    def __init__(self, alpha=None, beta=None, gamma=None, pf_curve='pf_linear', pf_interval=20, pf_std=0, name='fm'):

       # Failure Information
        self.name = name
        self.active = active
        self.pf_curve = pf_curve
        self.pf_interval = pf_interval
        self.pf_std = pf_std

        # Failure Distributions
        self.untreated = Distribution(alpha=alpha, beta=beta, gamma=gamma, name='untreated') #TODO drop this out and replace with something else for default
        self.init_dist = None
        self.pof = None  

        # Faiure Condition
        self.conditions = self.set_conditions(conditions)
        
        # Consequene of failure
        self.cof = Consequence()

        # Failre Mode state
        self.states = self.set_states(states)

        # Tasks
        self.tasks = self.set_tasks(tasks)

        # Prepare the failure mode
        self.calc_init_dist()  # TODO make this method based on a flag

        # Simulation details
        self.timeline = dict()
        self._timelines = dict()
        self._sim_counter = 0

        return


    # ************** Load Functions *****************

    def load(self,
        active=True,
        untreated=dict(),
        pf_curve='pf_linear',
        pf_interval=20,
        pf_std=0,
        name='fm', 
    ):

       # Failure Information
        self.name = name
        self.active = active
        self.pf_curve = pf_curve
        self.pf_interval = pf_interval
        self.pf_std = pf_std

        # Failure Distributions
        self.untreated = Distribution(untreated)
        self.init_dist = None
        self.pof = None  

        # Faiure Condition
        self.conditions = self.set_conditions(conditions)
        
        # Consequene of failure
        self.cof = Consequence()

        # Failre Mode state
        self.states = self.set_states(states)

        # Tasks
        self.tasks = self.set_tasks(tasks)

        # Prepare the failure mode
        self.calc_init_dist()  # TODO make this method based on a flag

        # Simulation details
        self.timeline = dict()
        self._timelines = dict()
        self._sim_counter = 0

    # ************** Set Functions *****************



    def set_untreated(self, untreated):
        self.untreated = untreated
        self.calc_init_dist()

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



    def calc_init_dist(self):  # TODO needs to get passed a condition and a pof
        """
        Convert the probability of failure into a probability of initiation
        """

        print("cacl init dist")
        # Super simple placeholder # TODO add other methods
        alpha = self.untreated.alpha
        beta = self.untreated.beta
        gamma = max(0, self.untreated.gamma - self.pf_interval)

        self.init_dist = Distribution(alpha=alpha, beta=beta, gamma=gamma, name ='init')


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

    def get_expected_pof(self):

        # TODO add a check if it has been simulated yet self.pof is None, self._timlines = None

        return self.pof

    # *************** Condition Loss *************** # TODO not updated to new method

    def measure_condition_loss(self):

        return self.condition.measure()

    def get_cond_loss(self):

        # New Condition at end of t

        return



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
                    conditions=self.conditions,
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
            self.timeline=dict(
                time=np.linspace(t_start, t_end, t_end - t_start + 1, dtype=int),
                intiation = np.full(t_end + 1, False),
                dectection = np.full(t_end + 1, False),
                failure = np.full(t_end + 1, False),
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
        if self.active:
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

    def save_timeline(self, i):
        self._timelines[i] = self.timeline
    
    def increment_counter(self):
        self._sim_counter = self._sim_counter + 1

    # ****************** Expected Methods  ************

    def expected_simple(self):
        """Returns all expected outcomes using a simple average formula"""

        #TODO strip out the values that don't matter

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
        #TODO general into expected event = 'failure', cumulative = True/False method
        t_failures = []

        t_min = self._timelines[0]['time'][0]
        t_max = self._timelines[0]['time'][-1] + 1

        # Get the time of first failure or age at failure
        for timeline in self._timelines.values():
            if timeline['failure'].any():
                t_failures.append(timeline['time'][timeline['failure']][0])
            else:
                t_failures.append(t_max)

        # Fit the weibull
        wbf = WeibullFitter()

        event_observed = (t_failures != t_max)

        wbf.fit(durations=t_failures, event_observed=event_observed)
        
        self.pof = Distribution(
            alpha = wbf.lambda_,
            beta = wbf.rho_,
        )

        return self.pof

    def _expected(self, timeline_key, first_event=True):
        #TODO general into expected event = 'failure', cumulative = True/False method
        t_events = []

        t_min = self._timelines[0]['time'][0]
        t_max = self._timelines[0]['time'][-1] + 1

        # Get the time of first failure or age at failure
        for timeline in self._timelines.values():
            if timeline['failure'].any():
                t_event = timeline['time'][timeline['failure']]
                if first_event:
                    t_events.append(t_event[0])
                else:
                    t_events.append(np.diff(np.append(t_min, t_event)))
            else:
                t_events.append(t_max)

        # Fit the weibull
        wbf = WeibullFitter()

        event_observed = (t_failures != t_max)

        wbf.fit(durations=t_failures, event_observed=event_observed)
        
        self.wbf = wbf

    def expected_condition(self):
        """Get the expected condition for a failure mode"""
        expected = dict()
        for cond_name in self.conditions.values():
            expected[cond_name] = np.array([self._timelines[x][cond_name] for x in self._timelines]).mean(axis=0)

        return expected

    def expected_condition_loss(self, stdev = 1):
        """Get the expected condition for a failure mode"""
        expected = dict()
        for cond_name, condition in self.conditions.items():
  
            ec = np.array([self._timelines[x][cond_name] for x in self._timelines])

            mean = condition.perfect - ec.mean(axis=0)
            sd = ec.std(axis=0)
            upper = mean + sd*stdev
            lower = mean - sd*stdev

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
            for details in erc.values():
                for task in details.values():
                    t_end = max(max(task['time'], default=t_start), t_end)

        df = pd.DataFrame(erc).T.apply(fill_blanks, axis = 1, args = (t_start, t_end))
        df.index.name = 'task'
        df_cost = df.explode('cost')['cost']
        df = df.explode('time')
        df['cost'] = df_cost

        # Add a cumulative cost
        df['cost_cumulative'] = df.groupby(by=['task'])['cost'].transform(pd.Series.cumsum)

        return df.reset_index()

    def expected_risk_cost(self, scaling=1):

        scaling = self._sim_counter

        profile = self._expected_cost(scaling=scaling)
        profile['risk'] = self._expected_risk(scaling=scaling)

        return profile

    def _expected_cost(self, scaling=1):

        task_cost = dict()
        
        # Get the costs causes by tasks
        for task_name, task in self.tasks.items():
            if task.active:
                task_cost[task_name] = task.expected_costs(scaling)

        return task_cost

    def _expected_risk(self, scaling=1): # TODO expected risk with or without replacement

        t_failures = []
        for timeline in self._timelines.values():
            if timeline['failure'].any():
                t_failures.append(np.argmax(timeline["failure"]))

        time, cost = np.unique(t_failures, return_counts = True)
        cost = cost * self.cof.get_cost() / scaling

        return dict(time=time, cost=cost)

    def expected_tasks(self):

        task_count = dict()

        for task_name, task in self.tasks.items():
            if task.active:
                task_count[task_name] = task.expected_counts(self._sim_counter)

        return task_count

    def expected_cost_df(self): # TODO delete. Backup only
        # TODO I think this is legacy?

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


        t_failures = []
        for timeline in self._timelines.values():
            if timeline['failure'].any():
                t_failures.append(np.argmax(timeline["failure"]))

        time, cost = np.unique(t_failures, return_counts = True)
        cost = cost * self.cof.get_cost() / self._sim_counter
        cost_cumulative = cost.cumsum()

        df = pd.DataFrame(
            dict(
                task='risk',
                time=time,
                cost=cost,
                cost_cumulative=cost_cumulative,
            )
        )

        df_tasks = pd.concat([df_tasks, df])

        return df_tasks

    # ****************** Reset Routines **************

    def reset_condition(self):

        # Reset conditions
        for condition in self.conditions.values():
            condition.reset()

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


    def plot_expected(self):

        fig, ax = plt.subplots()

        if not self.expected:

            ax.set_title("Expected outcome for %i simulations" %(len(self._timelines)))

            for key in list(self.timeline):
                ax.plot(self.expected['time'], self.expected[key])
                ax.fill_between(self.expected['time'], self.lower[key], self.upper[key], alpha=0.2)

        elif not self.timeline:

            self.plot_timeline()

        else:

            print("No timelines have been simulated")


    def update(self, dash_id, value, sep='-'):
        """Updates a the failure mode object using the dash componenet ID"""

        try:
            id_update(self, id_str=dash_id, value=value, sep=sep, children=(Condition, Distribution, Task))
            self.calc_init_dist()

        except:
            print('Invalid ID')

    def get_dash_ids(self, prefix="", sep='-'):
        """ Return a list of dash ids for values that can be changed"""

        prefix = prefix + "FailureMode" + sep + self.name + sep

        # Failure modes
        fm_ids = [prefix + param for param in ['active', 'pf_curve', 'pf_interval', 'pf_std']]

        # Failure Dist
        fd_ids = self.untreated.get_dash_ids(prefix=prefix)

        # Tasks
        task_ids = []
        for task in self.tasks.values():
            task_ids = task_ids + task.get_dash_ids(prefix=prefix + 'tasks' + sep)

        dash_ids = fm_ids + fd_ids + task_ids

        return dash_ids


    def get_objects(self,prefix="", sep = "-"):

        # Failure mode object
        prefix = prefix + 'FailureMode' + sep
        objects = [prefix + self.name]

        # Tasks objects
        prefix = prefix + self.name + sep
        objects = objects + [prefix + 'tasks' + sep + 'Task' + sep + task for task in self.tasks]

        return objects

    # ****************** Demonstration ***********

    def reset_demo(self):
        self.untreated=None
        self.conditions=None
        self.tasks = None
        self.states = None
        self.set_demo()

    def set_demo(self):

        

        if self.untreated is None:
            self.set_untreated(Distribution(alpha=50, beta=3, gamma=10, name='untreated'))
            # Prepare the failure mode
            self.calc_init_dist()

        if not self.conditions:
            self.set_conditions(
                dict(
                    wall_thickness=Condition(100, 0, "pf_linear", [-5], pf_interval=self.pf_interval, pf_std = self.pf_std),
                    external_diameter=Condition(100, 0, "pf_linear", [-2], pf_interval=self.pf_interval * 2, pf_std = self.pf_std),
                )
            )

        if not self.tasks:
            self.set_tasks(
                dict(
                    inspection=Inspection(t_interval=5, t_delay=10, name='inspection').set_default(),
                    on_condition_repair=OnConditionRepair(activity="on_condition_repair", name = 'on_condition_repair').set_default(),
                    on_condition_replacement=OnConditionReplacement(activity="on_condition_replacement", name = 'on_condition_replacement').set_default(),
                    immediate_replacement=ImmediateMaintenance(activity="immediate_replacement", name = 'immediate_replacement').set_default(),
                )
            )

        if not self.states:
            self.set_states(dict(initiation=False, detection=False, failure=False,))



        return self

if __name__ == "__main__":
    failure_mode = FailureMode()
    print("FailureMode - Ok")
