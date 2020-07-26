"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from matplotlib import pyplot as plt
from random import random, seed

from pof.condition import Condition
from pof.distribution import Distribution
from pof.consequence import Consequence
from pof.task import Inspection, Replace, Repair

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

seed(1)

class FailureMode: #Maybe rename to failure mode

    def __init__(self, alpha, beta, gamma):

        # Failure behaviour
        self.failure_dist = Distribution(alpha=50, beta=1.5, gamma=10)
        self.init_dist = None

        # Set the time period of interested # TODO Make this an input
        self.t = np.arange(0,101,1)

        self.pf_interval = 5 #TODO

        self.conditions = [
            Condition(100, 0, 'linear', [-5]),
        ]

        # Failure information
        self.t_fm = 0
        self.t_uptime = 0
        self.t_downtime = 0
        self.cof = Consequence() #TODO change to a consequence model
        self.pof = None #TODO

        # Failre Mode state
        self._initiated = False
        self._detected = False
        self._failed = False

        self.t_initiated = False #TODO

        # Tasks
        self.task_order = [1,2,3,4] # 'inspect', 'replace', repair' # todo
        
        self.tasks = [
            Replace(trigger = 'condition'),
            Inspection(trigger = 'time'),
        ]

        self.inspection = Inspection(trigger='time')
        self.corrective_maintenance = Replace()

        # Prepare the failure mode
        self.calc_init_dist()


        # State History
        self._history = dict(
            t_fm = [],
            _initiated = [],
            _detected = [],
            _failed = [],
            condition = [],
        
        ) #TODO fix this ugly beast up


        # kpis? #TODO
        # Cost and Value of current task? #TODO
        self.value = None #TODO

        return
    
    def calc_init_dist(self): #TODO needs to get passed a condition and a pof
        """
        Convert the probability of failure into a probability of initiation
        """

        # Super simple placeholder # TODO add other methods
        alpha = self.failure_dist.alpha
        beta = self.failure_dist.beta
        gamma = self.failure_dist.gamma - self.pf_interval

        self.init_dist = Distribution(alpha=alpha, beta=beta, gamma=gamma)

        return

    def calc_condition_loss(self, t_min, t_max):
        """

        """

        self.condition_loss = 1 - self.get_expected_condition(t_min, t_max)

        return 


    def get_expected_condition(self, t_min, t_max): #TODO retire?
        
        t_forecast = np.linspace(t_min, t_max, t_max-t_min+1, dtype = int)

        # Calculate the probability of initiation for the time period 
        prob_initiation = f_ti[t_forecast[1:]]

        # Add the probability after t_max onto the final row with perfect condition
        prob_not_considered = sf_i[t_max]
        prob_initiation = np.append(
            prob_initiation,
            prob_not_considered
        )

        # Scale to ignore the probability that occurs before t_min
        prob_initiation = prob_initiation / prob_initiation.sum()

        # Create a condition matrix of future condition
        deg_matrix = np.tril(np.full((deg_curve.size, deg_curve.size),100),-1) + np.triu(circulant(deg_curve).T)

        condition_outcomes = (deg_matrix.T * prob_initiation).T
        
        condition_mean = condition_outcomes.sum(axis=0)

        return True


        # Need to consider condition that has a probability of outcomes ()


    def get_probabilities(self, t_step):
        """
        Calculate the probabilities of being in a state after an interval
        """

        # Probability failure is initated
        p_i = self.get_p_initiation(t_step)
        
        # Probability of condition loss

        # Probability symptom is initiated # TODO

        # Probability failure is detected

        return p_i
        

    def is_initiated(self, t_step=None):
        """
        Return p
        """
        
        if self._initiated == True:
            return True

        else:

            p_i = self.init_dist.conditional_f(self.t_fm, self.t_fm + t_step)


    # *************** Condition Loss ***************

    def measure_condition_loss(self):

        return self.condition.measure()

    def get_cond_loss(self):

        # New Condition at end of t

        return
    
    def get_p_initiation(self, t_step): #TODO make a robust time step. (t_min, t_max, etc)

        if self._initiated == True:
            p_i = 1
        else:
            p_i = self.init_dist.conditional_f(self.t_fm, self.t_fm + t_step) #TODO update to actual time

        return p_i
    
    
    # ****************** Timeline ******************

    def sim_timeline(self, t_end, t_start=0):

        timeline = dict()

        # Get intiaition
        if self._initiated:
            tl_i = np.full(t_end, 1)
        else:
            tl_i = np.full(t_end, self._initiated)
            t_initiate = int(self.init_dist.sample())
            tl_i[:t_initiate] = 1

        timeline ['initiated'] = tl_i

        # Get condtiion
        tl_c = self.condition.get_condition_profile(t_start=t_start, t_stop=t_end)

        timeline [self.condition.name] = tl_c

        # Check time based tasks
        for task_name, task in self.tasks.items():

            if task.trigger == 'time': 
                tl_tt = task.event_schedule(t_end)
                events[task.activity] = tl_tt
        
        # Check detection
        if self._detected == True:
            tl_d = np.full(t_end, 1)
        else:
            tl_d = self.inspection.event_detection(t_end, timeline)

        timeline['detection'] = tl_d

        # Check failure

        if self._failed == True:
            tl_f = np.full(t_end, 1)
        else:
            
            for condition in self.conditions:
                tl_f = self.condition.check_failure_event(t_end)

        timeline['failure'] = tl_f


        return timeline
    


    # ****************** Simulate *******************

    def sim(self, t_step):

        # Check for initiation
        self.sim_initiation(t_step)

        # Check for condition
        if self._initiated:

            self.sim_condition(t_step) # TODO check for all condition for loop

            # Check for failure
            self.sim_failure(t_step)

            if self._failed:
                
                #Trigger corrective Maintenance
                self.sim_corrective_maintenance(t_step)

        # Check for detection TODO is this just a task
        self.sim_detection(t_step)

        # Check for tasks
            # Replace
            # Repair 

        # Increment time
        self.t_fm = self.t_fm + t_step

        # Record History
        self.record_history()

        return

    def sim_initiation(self, t_step):

        if self._initiated == False:

            p_i = self.get_p_initiation(t_step = t_step)
            
            if(random() < p_i):

                self._initiated = True
                self.t_initiated = self.t_fm

        return

    def sim_condition(self, t_step):

        # Simple method -> increment the condition
        return self.condition.sim(t_step)

    def sim_failure(self, t_step): #TODO failure doesn't need time

        # TODO add for loop and check all methods
        self._failed = self.condition.is_failed() #TODO or sytmpom or safety factor failure?

        return self._failed

    def sim_detection(self, t_step):

        det = self.inspection.sim_inspect(t_step, self.condition)

        if det == True:
            self._detected = True
          
    def sim_corrective_maintenance (self, t_step):
        """
        Currently stubbed as a replace task #TODO
        """
        # Complete the corrective maintenace task
        #if self.corrective_maintenance.is_triggered(self):
            


        return NotImplemented

    def sim_tasks(self, t_step):

        # Check time triggers, check condition triggers

        
        # Check time triggers
        for task in self.tasks:

            if task.trigger_type == 'time':
                
                # Check if the time has been triggered
                task.check_time(self.t_fm + t_step)



        # Check on condition triggers
        for task in self.tasks:
            
            task.check_trigger(self.t_fm + t_step)

        # Check value? trigger
        return NotImplemented


    def sim_history(self):

        nrows = len(self._history)
        fig, ax = plt.subplots(nrows=5, ncols=1)

        row = 0

        for field in  ["_initiated", "_detected", '_failed']:
            ax[row].step(self._history['t_fm'], self._history[field])
            ax[row].set_ylabel(field)
            
            row = row + 1

        for field in  ['condition']:
            ax[row].plot(self._history['t_fm'], self._history[field])
            ax[row].set_ylabel(field)
            
            row = row + 1

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

    def record_history(self):


        #for var in vars_record:
        #    self._history[var].append(self.)
        
        self._history['t_fm'].append(self.t_fm)
        self._history['_initiated'].append(self._initiated)
        self._history['_detected'].append(self._detected)
        self._history['_failed'].append(self._failed)
        self._history['condition'].append(self.condition.current())


    """

        # simple implementation first

            # Check for


        Check tasks are triggered

        Check tasks are executed

        Check tasks are in progress



        ** Check task groups

        ** Check value

        Execute tasks

    """