
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

        return

    def set_time_range(self):

        return