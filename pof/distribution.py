
import numpy as np
import scipy.stats as ss

class Distribution:

    # TODO Extend so that it works for all the common distributions

    def __init__(self, alpha, beta, gamma = None):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        X = np.arange(0,100,1)

        self.pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        self.cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        self.sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

        return

    def __str__(self):

        out = "Alpha = %s, Beta = %s, Gamma = %s" %(self.alpha, self.beta, self.gamma)
        
        return out

    def set_time_range(self):

        return

    def conditional_f(self, x_min, x_max):
        P_min = ss.weibull_min.sf(x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(x_max, self.beta, scale=self.alpha, loc=self.gamma)
        P = 1 - P_max / P_min

        return P

    def conditional_sf(self, x_min, x_max): # TODO Occa should this be total failure rate (cdf) or conditional failure

        P_min = ss.weibull_min.sf(x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(x_max, self.beta, scale=self.alpha, loc=self.gamma)
        P = P_max / P_min

        return P

    def sample(self, size=1):
        return ss.weibull_min.rvs(self.beta, scale=self.alpha, loc=self.gamma, size=size)


    def likelihood(self, x=None): # TODO not sure if we need this

        if x is None:
            #Return pdf
            return 1
        else:
            #Return 
            return 0
