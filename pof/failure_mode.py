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

from pof.degradation import Degradation
from pof.distribution import Distribution
from pof.consequence import Consequence

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

        self.degradation = Degradation(0,100) #TODO update with all the other options

        # Failure information
        self.t_fm = 0
        self.cof = Consequence() #TODO change to a consequence model
        self.pof = None #TODO

        # failure state
        self._initiated = False
        self._detected = False
        self._failed = False

        self.t_initiated = False #TODO

        # Symptoms TODO Change to dict or array of symptoms
        self.termites_present = False
        self.termites_detected = False

        # Tasks

        # Prepare the failure mode
        self.calc_init_dist()


        # State History
        self._history = dict(
            t_fm = [],
            _initiated = [],
            _detected = [],
            _failed = [],

        ) #TODO fix this ugly beast up


        # kpis? #TODO
        # Cost and Value of current task? #TODO
        self.value = None #TODO

        return
    
    def calc_init_dist(self): #TODO needs to get passed a degradation and a pof
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


    def get_expected_condition(self, t_min, t_max):
        
        """t_forecast = np.linspace(t_min, t_max, t_max-t_min+1, dtype = int)

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

        # Create a degradation matrix of future condition
        deg_matrix = np.tril(np.full((deg_curve.size, deg_curve.size),100),-1) + np.triu(circulant(deg_curve).T)

        condition_outcomes = (deg_matrix.T * prob_initiation).T
        
        condition_mean = condition_outcomes.sum(axis=0)"""

        return True


        # Need to consider degradation that has a probability of outcomes ()


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


    def p_failed(self, t_step=None):
        """
        """

        if self._failed == True:
            return 1

        else:

            return 0 # TODO add rest of options


    # *************** Condition Loss ***************

    def measure_condition_loss(self):

        return self.degradation.measure()

    def get_cond_loss(self):

        # New Condition at end of t

        return
    
    def get_p_initiation(self, t_step): #TODO make a robust time step. (t_min, t_max, etc)

        if self._initiated == True:
            p_i = 1
        else:
            p_i = self.init_dist.conditional_f(self.t_fm, self.t_fm + t_step) #TODO update to actual time

        return p_i
    

    # ****************** Simulate *******************

    def sim(self, t_step):

        # Check for initiation
        self.sim_initiation(t_step)

        # Check for degradation
        if self._initiated:

            # TODO check for all degradation for loop
            self.sim_degradation(t_step)

            # Check for failure
            self.sim_failure(t_step)

            if self._failed:
                
                #Trigger corrective Maintenance
                self.corrective_maintenance()

        # Check for detection TODO is this just a task
        
        #self.sim_detection(t_step)

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

    def sim_degradation(self, t_step):

        # Simple method -> increment the condition
        return self.degradation.sim(t_step)

    def sim_failure(self, t_step):

        # TODO add for loop and check all methods
        self._failed = self.degradation.limit_reached() #TODO or sytmpom or safety factor failure?

        return self._failed

    def sim_tasks(self, t_step):

        # Check if task is triggered
        # Implement action
        return


    def sim_history(self):

        nrows = len(self._history)
        fig, ax = plt.subplots(nrows=5, ncols=1)

        row = 0
        for field in self._history:

                ax[row].plot(self._history[field])
                ax[row].set_ylabel(field)

            row = row + 1


    def corrective_maintenance(self):

        return

    def record_history(self):

        vars_record = ['t_fm', "_initiated", "_detected", '_failed']

        #for var in vars_record:
        #    self._history[var].append(self.)
        
        self._history['t_fm'].append(self.t_fm)
        self._history['_initiated'].append(self._initiated)
        self._history['_detected'].append(self._detected)
        self._history['_failed'].append(self._failed)
        self._history['_failed'].append(self._failed)