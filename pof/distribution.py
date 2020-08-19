import numpy as np
import scipy.stats as ss


class Distribution:

    # TODO Extend so that it works for all the common distributions

    # Default Values
    _ALPHA = 100
    _BETA = 1.5
    _GAMMA = 10

    def __init__(self, alpha=None, beta=None, gamma=None):

        self.alpha = alpha if alpha is not None else Distribution._ALPHA
        self.beta = beta if beta is not None else Distribution._BETA
        self.gamma = gamma if gamma is not None else Distribution._GAMMA

        X = np.arange(0, 100, 1)

        self.pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        self.cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        self.sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

        return

    def __str__(self):

        out = "Alpha = %s, Beta = %s, Gamma = %s" % (self.alpha, self.beta, self.gamma)

        return out

    def params(self):
        params = dict(
            alpha = self.alpha,
            beta = self.beta,
            gamma = self.gamma,
        )
        return params

    def set_time_range(self):

        return

    def conditional_f(self, x_min, x_max):
        P_min = ss.weibull_min.sf(x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(x_max, self.beta, scale=self.alpha, loc=self.gamma)
        P = 1 - P_max / P_min

        return P

    def conditional_sf(
        self, x_min, x_max
    ):  # TODO Occa should this be total failure rate (cdf) or conditional failure

        P_min = ss.weibull_min.sf(x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(x_max, self.beta, scale=self.alpha, loc=self.gamma)
        P = P_max / P_min

        return P

    def sample(self, size=1):
        return ss.weibull_min.rvs(
            self.beta, scale=self.alpha, loc=self.gamma, size=size
        )

    def likelihood(self, x=None):  # TODO not sure if we need this

        if x is None:
            # Return pdf
            return 1
        else:
            # Return
            return 0


    def get_dash_id(self, prefix=''):

        return [prefix + param for param in ['alpha', 'beta', 'gamma']]

    def dash_update(self, dash_id, value):
        """Updates a the distribution object using the dash component ID"""

        try:

            next_id = dash_id.split('-')[0]
            self.__dict__[next_id] = value

        except:
            print("Invalid dash component %s" %(dash_id))

if __name__ == "__main__":
    distribution = Distribution()
    print("Distribution - Ok")
