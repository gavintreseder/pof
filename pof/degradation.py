"""

Author: Gavin Treseder
"""

# ************ Packages ********************
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import circulant

#import Distribution
#TODO move t somewhere else
#TODO create better constructors https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

class Degradation:

    def __init__(self, alpha, beta, gamma):

        # Move to this stuff to a Distribution class latre
        self.pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        self.cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)

        # Get expected degradation_durve
        self.t = np.arange(0,101,1)
        
        self.degradation_curve = self.cdf
        
        return
    

    def get_dist_data(self, alpha, beta, gamma):
        
        

        return True

    def condition_loss(self, t_min, t_max):
        """

        """

        # Get possible condtion degradation



        return True