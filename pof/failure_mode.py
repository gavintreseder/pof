"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant
from random import random
from random import seed

from pof.degradation import Degradation
from pof.distribution import Distribution

#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

seed(1)

class FailureMode: #Maybe rename to failure mode

    def __init__(self, alpha, beta, gamma):

        # Failure behaviour
        self.failure_dist = Distribution(alpha=50, beta=1.5, gamma=10)

        # Set the time period of interested # TODO Make this an input
        self.t = np.arange(0,101,1)

        self.pf_interval = 5 #TODO

        self.degradation = Degradation(0,100) #TODO update with all the other options

        # Failure information
        self.t_fm = 0

        # failure state
        self.initiated = False
        self.detected = False

        self.t_initiated = False #TODO

        # Symptoms TODO Change to dict or array of symptoms
        self.termites_present = False
        self.termites_detected = False

        # Tasks

        # Prepare the failure mode
        self.calc_init_dist()


        # State History


        return
    
    def calc_init_dist(self): #TODO needs to get passed a degradation and a pof
        """
        Convert the probability of failure into a probability of initiation
        """

        # Super simple placeholder
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
        

    def sim_step(self, t_step):
        """
        Calculate the expected state of an asset after an interval
        """

        # Check if it has initiated 
        self.sim_initiation(t_step)

        # Check if symptom present

        # Check condition loss

        # Check if symptom detected

        # Check condition measured


        # Initation - check for initiation

            # Sytmptoms
            # Condition Loss

            # Initiation Flag
        
        # Inspection - check symptoms and condition

            # Detect Symptoms
            # Measure Condition

            # Detection Flag


        # Task - if detected, check task triggers

            # Do task action based on trigger (condition windows)


        # Failure - check for failure


        # Incremenet failure mode age
        self.t_fm = self.t_fm + t_step

    def check_for_initiation(self, t_step):
        
        if self.initated == True:
            return True
    
        else:
            p_i = self.init_dist.likelihood(self.t_fm, self.t_fm + t_step)


    # *************** Condition Loss ***************

    def measure_condition_loss(self):

        return self.expected_condition

    def get_cond_loss(self):

        # New Condition at end of t

        return
    
    def get_p_initiation(self, t_step): #TODO make a robust time step. (t_min, t_max, etc)

        if self.initiated == True:
            p_i = 1
        else:
            p_i = self.init_dist.conditional_f(self.t_fm, self.t_fm + t_step) #TODO update to actual time

        return p_i
    

    # ****************** Simulate *******************

    def sim_initiation(self,t_step):

        if self.initiated == False:

            p_i = self.get_p_initiation(t_step = t_step)
            
            if(random() < p_i):

                self.initiated = True
                self.t_initiated = self.t_fm


    def sim_condition_loss(self, t_step):

        # Simple method -> increment the condition
        return self.degradation.expected(t_step - self.t_fm)

