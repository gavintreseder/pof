"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss

from tqdm import tqdm
from lifelines import WeibullFitter

from failure_mode import FailureMode
from distribution import Distribution
from helper import fill_blanks

#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
#TODO create get, set, del and add methods

DEFAULT_ITERATIONS = 100

class Component():
    """
        Parameters:

        Methods:
            
    """

    def __init__(self):
        
        # Object parameters
        self.active = True

        # Initial parameters
        self.age = 0
        self.age_last_insp = 0

        # Link to other componenets
        self._parent_id = NotImplemented
        self._children_id = NotImplemented
        
        # Parameter
        self.name = 'comp' # TODO NotImplemented fully
        self.fm = dict()
        self.conditions = dict()

        # Simulation traking
        self._sim_counter = 0

    # ****************** Load data ******************

    def load(self):
        # Load Failure Modes
        # Load asset information
        NotImplemented


    # ****************** Set data ******************
    
    def set_params(self, name=None):

        self.name = name

        return NotImplemented

    def mc(self, t_end, t_start, n_iterations):
        """ Complete MC simulation and calculate all the metrics for the component"""

        return NotImplemented

    # ****************** Timeline ******************

    def mc_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times"""
        self.reset()  # TODO ditch this ... Check why this was in failure mode

        for i in tqdm(range(n_iterations)):
            self.sim_timeline(t_end=t_end, t_start=t_start)

            for fm in self.fm.values():
                fm.save_timeline(i)
    
    def sim_timeline(self, t_end, t_start=0):
        """ Simulates the timelines for all failure modes attached to this copmonent"""

        # Initialise the failure modes
        self.init_timeline(t_start=t_start, t_end = t_end)

        t_now = t_start

        while t_now < t_end:

            t_next, next_fm_tasks = self.next_tasks(t_now)

            self.complete_tasks(t_next, next_fm_tasks)

            t_now = t_next + 1

        self.increment_counter()
        self.reset_for_next_sim()

    def init_timeline(self, t_end, t_start=0):
        """ Initilialise the timeline"""
        for fm in self.fm.values():
            fm.init_timeline(t_start=t_start, t_end=t_end)

    def next_tasks(self, t_now):
        """
        Returns a dictionary with the failure mode triggered
        """
        # TODO make this more efficent
        # TODO make this work if no tasks returned. Expect an error now

        # Get the task schedule for next tasks
        task_schedule = dict()
        for fm_name, fm in self.fm.items():

            t_now, task_names = fm.next_tasks(t_start=t_now)

            if t_now in task_schedule:
                task_schedule[t_now][fm_name] = task_names
            else:
                task_schedule[t_now] = dict()
                task_schedule[t_now][fm_name] = task_names

        t_now = min(task_schedule.keys())

        return t_now, task_schedule[t_now]

    def complete_tasks(self, t_now, fm_tasks):
        """Complete any tasks in the dictionary fm_tasks at t_now"""

        # TODO add logic around all the different ways of executing
        # TODO check task groups
        # TODO check value?
        # TODO add task impacts

        for fm_name, task_names in fm_tasks.items():
            system_impact = self.fm[fm_name].complete_tasks(t_now, task_names)

            """ if system_impact != False:

                self.reset_condition()
                #TODO add impact on other systems for complex systems"""


    def increment_counter(self):
        self._sim_counter = self._sim_counter + 1

        for fm in self.fm.values():
            fm.increment_counter()

    # ****************** Expected ******************

    # TODO RUL

    def expected(self):

        # Run expected method on all failure modes

        # Run 
        NotImplemented

    # PoF

    def expected_cdf(self, t_start=0, t_end=100):

        sf = self.expected_sf(t_start, t_end)
        
        cdf = dict()

        for fm in sf:
            cdf[fm] = 1 - sf[fm]

        return cdf

    def expected_sf(self, t_start=0, t_end=100):
        

        # Calcuate the failure rates for each failure mode
        sf=dict()

        for fm_name, fm in self.fm.items():
            pof = fm.expected_pof()

            sf[fm_name] = pof.sf(t_start, t_end)

        # Treat the failure modes as a series and combine together
        sf['all'] = np.array([fm.sf(t_start, t_end) for fm in self.fm.values()]).prod(axis=0)

        # TODO Fit a new Weibull for the new failure rate....

        return sf


    def expected_risk_cost_df(self, t_start = 0, t_end=100):
        """ A wrapper for expected risk cost that returns a dataframe"""
        erc = self.expected_risk_cost()

        df = pd.DataFrame().from_dict(erc, orient='index')
        df.index.name='failure_mode'
        df = df.reset_index().melt(id_vars = 'failure_mode', var_name='task')
        df = pd.concat([df.drop(columns=['value']),df['value'].apply(pd.Series)], axis=1)
        df = df.apply(fill_blanks, axis=1, args=(t_start,t_end))
        df_cost = df.explode('cost')['cost']
        df = df.explode('time')
        df['cost'] = df_cost

        # Add a cumulative cost
        df['cost_cumulative'] = df.groupby(by=['failure_mode', 'task'])['cost'].transform(pd.Series.cumsum)

        return df

    def expected_risk_cost(self):

        ec = dict()
        for fm_name, fm in self.fm.items():
            ec[fm_name] = fm.expected_risk_cost()

        return ec

    # Cost

    # ****************** Reset ******************

    def reset_condition(self):

        for fm in self.fm.values():
            fm.reset_condition()

    def reset_for_next_sim(self):
        """ Reset parameters back to the initial state"""

        for fm in self.fm.values():
            fm.reset_for_next_sim()


    def reset(self):
        """ Reset all parameters back to the initial state and reset sim parameters"""

        # Reset failure modes
        for fm in self.fm.values():
            fm.reset()

        # Reset counters
        self._sim_counter = 0

    # ****************** Interface ******************

    def dash_update(self, dash_id, value, sep='-'):
        """Updates a the failure mode object using the dash componenet ID"""

        try:
            
            next_id = dash_id.split(sep)[0]
        
            # Check if the next component is a param of 
            if next_id in ['active']:

                self.active = value

            elif next_id == 'failure_mode':

                dash_id = dash_id.replace(next_id + sep, "")
                fm_name= dash_id.split(sep)[0]
                dash_id = dash_id.replace(fm_name + sep, "")
                self.fm[fm_name].dash_update(dash_id, value)

        except:

            print("Invalid dash component %s" %(dash_id))


    def get_dash_ids(self, prefix="", sep='-'):
        """ Return a list of dash ids for values that can be changed"""

        # Component
        comp_ids = [prefix + param for param in ['active']]

        # Tasks
        fm_ids = []
        for fm_name, fm in self.fm.items():
            fm_prefix = prefix + 'failure_mode' + sep + fm_name + sep
            fm_ids = fm_ids + fm.get_dash_ids(prefix=fm_prefix)

        dash_ids = comp_ids + fm_ids

        return dash_ids

    # ****************** Demonstration parameters ******************

    def set_demo(self):
        """ Loads a demonstration data set if no parameters have been set already"""

        if not self.fm:
            self.fm = dict(
                fast_aging = FailureMode(alpha=50, beta=2, gamma=20).set_demo(),
                slow_aging = FailureMode(alpha=100, beta=1.5, gamma=20).set_demo(),
                random = FailureMode(alpha=500, beta=1, gamma=0).set_demo(),
            )
    
        return self

    def reset_demo(self):
        """ Loads a demonstration data set for all parameters"""
        self.fm = dict()

        self.set_demo()


if __name__ == "__main__":
    component = Component()
    print("Component - Ok")