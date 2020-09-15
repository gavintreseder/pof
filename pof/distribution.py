import numpy as np
import scipy.stats as ss


if __package__ is None or __package__ == '':
    from helper import id_update
else:
    from pof.helper import id_update


class Distribution:

    # TODO Extend so that it works for all the common distributions

    def __init__(self, alpha=100, beta=1, gamma=0, name='dist', *args, **kwargs):

        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        #self.pdf = ss.weibull_min.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        #self.cdf = ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        #self.sf = ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)


    @classmethod
    def from_dict(cls, details=None):
        try:
            fm = cls(**details)
        except:
            raise ValueError(
                "Error loading %s data from dictionary" % (cls.__name__))
        return fm

    def __str__(self):

        out = "Alpha = %s, Beta = %s, Gamma = %s" % (
            self.alpha, self.beta, self.gamma)

        return out

    def load(self, name='dist', **kwargs):
        self.last_name = name
        self.alpha = kwargs.get('alpha')
        self.beta = kwargs.get('beta')
        self.gamma = kwargs.get('gamma')

        return self



    def params(self):
        params = dict(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        return params

    def sf(self, t_start, t_end):
        X = np.linspace(t_start, t_end, t_end - t_start + 1)
        return ss.weibull_min.sf(X, self.beta, scale=self.alpha, loc=self.gamma)

    def cdf(self, t_start, t_end):
        X = np.linspace(t_start, t_end, t_end - t_start + 1)
        return ss.weibull_min.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)

    def set_time_range(self):

        return

    def conditional_f(self, x_min, x_max):
        P_min = ss.weibull_min.sf(
            x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(
            x_max, self.beta, scale=self.alpha, loc=self.gamma)
        P = 1 - P_max / P_min

        return P

    def conditional_sf(
        self, x_min, x_max
    ):  # TODO Occa should this be total failure rate (cdf) or conditional failure

        P_min = ss.weibull_min.sf(
            x_min, self.beta, scale=self.alpha, loc=self.gamma)
        P_max = ss.weibull_min.sf(
            x_max, self.beta, scale=self.alpha, loc=self.gamma)
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

    def get_dash_ids(self, prefix='', sep='-'):

        prefix = prefix + 'Distribution' + sep + self.name + sep
        return [prefix + param for param in ['alpha', 'beta', 'gamma']]

    def update(self, dash_id, value, sep='-'):
        """Updates a the distrubtion object using the dash componenet ID"""

        try:

            id_update(self, id_str=dash_id, value=value, sep=sep)

        except:
            print('Invalid ID')

    def load_demo(self, scenario=None):

        if scenario is None:

            data = dict(
                name="slow_aging",
                alpha=100,
                beta=2,
                gamma=10,
            )

        self.load(data)

    def csf(self, t_start, t_end):
        t_interval = np.arange(t_start, t_end + 1, 1)

        P_interval = ss.weibull_min.sf(
            t_interval, self.beta, scale=self.alpha, loc=self.gamma)

        P = P_interval / P_interval[0]

        return P

    def cff(self, t_start, t_end):
        t_interval = np.arange(t_start, t_end + 1, 1)

        P_interval = ss.weibull_min.sf(
            t_interval, self.beta, scale=self.alpha, loc=self.gamma)

        P = 1 - P_interval / P_interval[0]

        return P

    def update_from_dict(self, dict_data):

        for key, value in dict_data.items():

            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                print("ERROR: Cannot update \"%s\" from dict" %
                      (self.__class__.__name__))


"""class Dataset:

    self.early_life = dict(
        name = "infant_mortality",
        alpha = 10000,
        beta = 0.5,
        gamma = 0,
    )

    self.random = dict(
        name = "random",
        alpha = 100,
        beta = 1,
        gamma = 0,
    )

    self.slow_aging = dict(
        name = "slow_aging",
        alpha = 100,
        beta = 2,
        gamma = 10,
    )

    self.fast_aging = dict(
        name = "fast_aging",
        alpha = 100,
        beta = 3.5,
        gamma = 10,
    )


    def distribution(self, scenario):

        return Distribtuion(cls.__dict__[scenario])

class Demo:

    early_life = Distribution(Dataset.early_life)
    random = Distribution(Dataset.random)
    slow_aging = Distribution(Dataset.slow_aging)
    fast_aging = Distribution(Dataset.fast_aging)"""

if __name__ == "__main__":

    distribution = Distribution()

    print("Distribution - Ok")
