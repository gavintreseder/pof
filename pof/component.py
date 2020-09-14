"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss

from tqdm import tqdm
from lifelines import WeibullFitter

if __package__ is None or __package__ == '':
    from failure_mode import FailureMode
    from condition import Condition
    from distribution import Distribution
    from helper import fill_blanks, id_update
    from indicator import Indicator, ConditionIndicator, PoleSafetyFactor
    import demo as demo

else:
    from pof.failure_mode import FailureMode
    from pof.condition import Condition
    from pof.distribution import Distribution
    from pof.helper import fill_blanks, id_update
    from pof.indicator import Indicator, ConditionIndicator, PoleSafetyFactor
    import pof.demo as demo
    
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
#TODO create get, set, del and add methods

DEFAULT_ITERATIONS = 100

class Component():
    """
        Parameters:

        Methods:
            
    """

    def __init__(self,
        active = True,
        name = 'comp',
        indicator = dict(),
        fm = dict(),
        *args,
        **kwargs,

    ):
        
        # Object parameters
        self.active = active

        # Initial parameters
        self.age = 0
        self.age_last_insp = 0

        # Link to other componenets
        self._parent_id = NotImplemented
        self._children_id = NotImplemented
        
        # Parameter
        self.name = name # TODO NotImplemented fully
        self.indicator = dict()
        
        # Failure Modes
        self.fm = dict()
        self.set_failure_mode(fm)

        # Trial for indicator
        self.indicator['safety_factor'] = PoleSafetyFactor(component=self)
        self.indicator['slow_degrading'] = Condition.load(demo.condition_data['slow_degrading']) # TODO fix this call
        self.indicator['fast_degrading'] = Condition.load(demo.condition_data['fast_degrading']) # TODO fix this call

        #TODO link failure_modes to indicators

        # Simulation traking
        self._sim_counter = 0

    # ****************** Load data ******************

    @classmethod
    def load(cls, details=None):
        try:
            comp = cls.from_dict(details)
        except:
            comp = cls()
            print("Error loading Component data")
        return comp

    @classmethod
    def from_dict(cls, details=None):
        try:
            comp = cls(**details)
        except:
            comp = cls()
            print("Error loading Component data from dictionary")
        return comp



    def load_asset_data(self, *args, **kwargs):

        self.info = dict(
            pole_load = 10,
            pole_strength = 20,
        )

        # Set perfect indicator values?? TODO

        # Set indicators
        for indicator in self.indicator.values():

            # Set perfect
            x = 2
            # Set current
            NotImplemented


    def set_indicators(self, indicators):
        """
        Takes a dictionary of Indicator objects or indicator data and sets the component indicators
        """

        for name, indicator in indicators.items():
        
            # Load a condition object
            if isinstance(indicator, Condition):
                self.indicator[name] = indicator
            
            # Create a condition object
            elif isinstance(indicator, dict): #TODO add methods for different 
                self.indicator[name] = Indicator(**indicator)

    def set_failure_mode(self, failure_modes):
        """
        Takes a dictionary of FailureMode objects or FailureMode data and sets the component failure modes
        """

        for name, fm in failure_modes.items():
        
            # Load a condition object
            if isinstance(fm, FailureMode):
                self.fm[name] = fm
            
            # Create a condition object
            elif isinstance(fm, dict):
                self.fm[name] = FailureMode().load(fm)        



    # ****************** Set data ******************


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

            self._sim_counter = self._sim_counter + 1
    
    def sim_timeline(self, t_end, t_start=0):
        """ Simulates the timelines for all failure modes attached to this component"""

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

    def next_tasks(self, t_start):
        """
        Returns a dictionary with the failure mode triggered
        """
        # TODO make this more efficent
        # TODO make this work if no tasks returned. Expect an error now

        # Get the task schedule for next tasks
        task_schedule = dict()
        for fm_name, fm in self.fm.items():

            t_next, task_names = fm.next_tasks(t_start=t_start)

            if t_next in task_schedule:
                task_schedule[t_next][fm_name] = task_names
            else:
                task_schedule[t_next] = dict()
                task_schedule[t_next][fm_name] = task_names

        t_next = min(task_schedule.keys())

        return t_next, task_schedule[t_next]

    def complete_tasks(self, t_next, fm_tasks):
        """Complete any tasks in the dictionary fm_tasks at t_now"""

        # TODO add logic around all the different ways of executing
        # TODO check task groups
        # TODO check value?
        # TODO add task impacts

        for fm_name, task_names in fm_tasks.items():
            system_impact = self.fm[fm_name].complete_tasks(t_next, task_names)

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

    def expected_untreated(self, t_start=0, t_end=100):

        sf=dict(
            all = np.full((t_end - t_start + 1), 1)
        )
        
        for fm in self.fm.values():
            if fm.active:
                sf[fm.name] = fm.untreated.sf(t_start=t_start,t_end=t_end)
                sf['all'] = sf['all'] * sf[fm.name]

        # Treat the failure modes as a series and combine together
        cdf = {fm : 1 - sf for fm, sf in sf.items()}

        return cdf

    def expected_pof(self, t_start=0, t_end=100):

        sf = self.expected_sf(t_start, t_end)
        
        cdf = dict()

        for fm in sf:
            cdf[fm] = 1 - sf[fm]

        return cdf

    def expected_sf(self, t_start=0, t_end=100):
        
        # Calcuate the failure rates for each failure mode
        sf=dict(
            all = np.full((t_end - t_start + 1), 1)
        )
   
        for fm_name, fm in self.fm.items():
            if fm.active:
                pof = fm.expected_pof()

                sf[fm_name] = pof.sf(t_start, t_end)

                sf['all'] = sf['all'] * sf[fm_name]
        # Treat the failure modes as a series and combine together
        #sf['all'] = np.array([fm.sf(t_start, t_end) for fm in self.fm.values()]).prod(axis=0)

        # TODO Fit a new Weibull for the new failure rate....

        return sf


    def expected_risk_cost_df(self, t_start = 0, t_end=None):
        """ A wrapper for expected risk cost that returns a dataframe"""
        erc = self.expected_risk_cost()

        if t_end == None:
            t_end = t_start
            for details in erc.values():
                for task in details.values():
                    t_end = max(max(task['time'], default=t_start), t_end)

        df = pd.DataFrame().from_dict(erc, orient='index')
        df.index.name='failure_mode'
        df = df.reset_index().melt(id_vars = 'failure_mode', var_name='task')
        df = pd.concat([df.drop(columns=['value']),df['value'].apply(pd.Series)], axis=1)[['failure_mode', 'task', 'time', 'cost']].dropna()
        df = df.apply(fill_blanks, axis=1, args=(t_start,t_end))
        df_cost = df.explode('cost')['cost']
        df = df.explode('time')
        df['cost'] = df_cost

        # Add a cumulative cost
        df['cost_cumulative'] = df.groupby(by=['failure_mode', 'task'])['cost'].transform(pd.Series.cumsum)

        return df

    def expected_risk_cost(self):

        # Add scaling
    
        ec = dict()
        for fm_name, fm in self.fm.items():
            if fm.active:
                ec[fm_name] = fm.expected_risk_cost()

        return ec

    def expected_indicators(self):
        
        ei = dict()

        for indicator in self.indicator.values():
            
            indicator.expected()

        NotImplemented
        return ei


    def expected_condition(self, stdev=1): #TODO make work for all condition levels
        
        """ec = self.expected_condition_loss()
        for c in ec:
            ec[c] = 100 - ec[c]"""

        expected = dict()

        for fm in self.fm.values():
            # Get the expected condition loss for the failure mode

            if fm.active:
                for cond_name, condition in fm.conditions.items():
    
                    ec = np.array([fm._timelines[x][cond_name] for x in fm._timelines])

                    if cond_name in expected:
                        expected[cond_name] = expected[cond_name] + ec
                    else:
                        expected[cond_name] = ec

        for cond_name, ecl in expected.items():
            mean = ecl.mean(axis=0)
            sd = ecl.std(axis=0)
            upper = mean + sd*stdev
            lower = mean - sd*stdev

            upper[upper > condition.perfect] = condition.perfect
            lower[lower < condition.failed] = condition.failed

            expected[cond_name] = dict(
                lower=lower,
                mean=mean,
                upper=upper,
            )

        return expected

    def expected_condition_loss_legacy(self):
        """Get the expected condition loss for a component"""
        expected = dict()

        for fm in self.fm.values():
            # Get the expected condition loss for the failure mode
            ec = fm.expected_condition_loss()
            for c in ec:
                if c in expected:
                    expected[c]['mean'] = expected[c]['mean'] + ec[c]['mean']
                    #TODO change this to a pooled variance method
                    expected[c]['sd'] = (expected[c]['sd']**2 + ec[c]['sd']**2)**0.5
                    expected[c]['lower'] = expected[c]['mean'] - expected[c]['sd']
                    expected[c]['upper'] = expected[c]['mean'] + expected[c]['sd']
                else:
                    expected[c] = ec[c]

        return expected

    def expected_condition_loss(self, stdev=1):
        """Get the expected condition loss for a component"""
        # TODO move this back out so that condition works as an indpendent class
        expected = dict()

        for fm in self.fm.values():
            # Get the expected condition loss for the failure mode

            if fm.active:
                for cond_name, condition in fm.conditions.items():
    
                    ec = condition.perfect - np.array([fm._timelines[x][cond_name] for x in fm._timelines])

                    if cond_name in expected:
                        expected[cond_name] = expected[cond_name] + ec
                    else:
                        expected[cond_name] = ec

        for cond_name, ecl in expected.items():
            mean = ecl.mean(axis=0)
            sd = ecl.std(axis=0)
            upper = mean + sd*stdev
            lower = mean - sd*stdev

            upper[upper > condition.perfect] = condition.perfect
            lower[lower < condition.failed] = condition.failed

            expected[cond_name] = dict(
                lower=lower,
                mean=mean,
                upper=upper,
            )

        return expected

    # Cost

    # ****************** Optimal? ******************

    def expected_inspection_interval(self, t_max, t_min=0, step=1, n_iterations=100):
        #TODO add an optimal onto this
        rc = dict()
        self.reset()

        for i in range(max(1, t_min), t_max, step):

            # Set t_interval
            for fm in self.fm.values():
                if 'inspection' in list(fm.tasks):
                    fm.tasks['inspection'].t_interval = i
            
            self.mc_timeline(t_end=100, n_iterations=n_iterations)

            rc[i] = self.expected_risk_cost_df().groupby(by=['task'])['cost'].sum()
            rc[i]['inspection_interval'] = i

            # Reset component
            self.reset()

        df = pd.DataFrame().from_dict(rc, orient='index').rename(columns={'risk':'risk_cost'})
        df['direct_cost'] = df.drop(['inspection_interval', 'risk_cost'], axis=1).sum(axis=1)
        df['total'] = df['direct_cost'] + df['risk_cost']
        
        return df

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

    def update(self, dash_id, value, sep='-'):
        """Updates the component class using the dash componenet ID"""

        try:
           id_update(self, id_str=dash_id, value=value, sep=sep, children=[FailureMode])

        except:
            print('Invalid ID')


    def get_dash_ids(self, prefix="", sep='-'):
        """ Return a list of dash ids for values that can be changed"""

        # Component
        prefix = prefix + 'Component' + sep + self.name + sep
        comp_ids = [prefix + param for param in ['active']]

        # Tasks
        fm_ids = []
        for fm in self.fm.values():
            fm_ids = fm_ids + fm.get_dash_ids(prefix=prefix + 'fm' + sep)

        dash_ids = comp_ids + fm_ids

        return dash_ids

    def get_objects(self,prefix="", sep = "-"):

        prefix = prefix + "Component" + sep
        objects = [prefix + self.name]

        prefix = prefix + self.name + sep

        for fms in self.fm.values():
            objects = objects + fms.get_objects(prefix = prefix + 'fm' + sep)

        return objects

    # ****************** Demonstration parameters ******************

    def set_demo(self):
        """ Loads a demonstration data set if no parameters have been set already"""

        if not self.fm:
            self.fm = dict(
                early_life = FailureMode().load(demo.failure_mode_data['early_life']),
                random = FailureMode().load(demo.failure_mode_data['random']),
                slow_aging = FailureMode().load(demo.failure_mode_data['slow_aging']),
                fast_aging = FailureMode().load(demo.failure_mode_data['fast_aging']),
            )
    
        # Trial for indicator
        self.indicator['safety_factor'] = PoleSafetyFactor(component=self)
        self.indicator['slow_degrading'] = Condition(**demo.condition_data['slow_degrading']) # TODO fix this call
        self.indicator['fast_degrading'] = Condition(**demo.condition_data['fast_degrading']) # TODO fix this call

        """for fm in self.fm.values():
            fm.set_conditions(dict(
                slow_degrading = self.indicator['slow_degrading'],
                fast_degrading = self.indicator['fast_degrading'],
            ))"""


        return self

    def set_pole(self):

        NotImplemented



if __name__ == "__main__":
    component = Component()
    print("Component - Ok")