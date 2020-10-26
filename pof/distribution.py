from dataclasses import dataclass, field

import numpy as np
import scipy.stats as ss

# Change the system path when this is run directly
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.load import Load
from pof.helper import id_update, str_to_dict
from config import config

cf = config["Distribution"]

# TODO Extend so that it works for all the common distributions

class Manager(object):

    """

    #TODO Maybe move the set and update methods on the manager objects?

    """
    def set_obj(self, attr, d_type, value):
        """

        value = {'tasks':{'inspection':{'t_interval':10}}}
        """

        try:
            if value is None:
                setattr(self, attr, dict())

            # Add the value to the dictionary if it is an object of that type
            elif isinstance(value, d_type):
                getattr(self, attr)[value.name] = value

            # Check if the input is an iterable
            elif isinstance(value, Iterable):

                # Create an object from the dict
                if all([hasattr(d_type, attr) for attr in value]):
                    new_object = d_type.from_dict(value)
                    getattr(self, attr)[new_object.name] = new_object

                # Create an object from the dict of dict/objects
                else:
                    for key, val in value.items():

                        if isinstance(val, d_type):
                            getattr(self, attr)[val.name] = val

                        else:
                            new_object = d_type.from_dict(val)
                            getattr(self, attr)[new_object.name] = new_object

            else:
                raise ValueError

class DistributionManager(dict):
    def __setitem__(self, item, value):

        #untreated = self.get("untreated", None)
        untreated_hash = hash(self.get("untreated", None))

        # value = set_obj
        # item = value.name

        super(DistributionManager, self).__setitem__(item, value)

        if untreated_hash != hash(self.get("untreated", None)):
            init = Distribution.from_pf_interval(
                self.dists["untreated"], self.pf_interval
            )
            self.__setattr__('init', init)
            


@dataclass()
class DistributionData(Load):
    """
    A class that contains the data for the Distribution object.
    """

    alpha: int = field(default_factory=lambda: cf.getint("alpha"))
    beta: int = field(default_factory=lambda: cf.getint("beta"))
    gamma: int = field(default_factory=lambda: cf.getint("gamma"))
    name: str = field(default_factory=lambda: cf.get("name"))


class Distribution(DistributionData):

    """
    Usage:

    First import Distribution using the distribution package

        >>> from distribution import Distribution

    Create a Distribution object

        >>> Distribution()
        Distribution(name=...)

    """

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

    def get_dash_ids(self, prefix="", sep="-"):

        prefix = prefix + self.name + sep
        return [prefix + param for param in ["alpha", "beta", "gamma"]]

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
            t_interval, self.beta, scale=self.alpha, loc=self.gamma
        )

        P = P_interval / P_interval[0]

        return P

    @classmethod
    def from_pf_interval(cls, dist, pf_interval):
        """
        Returns a distribution that has been adjusted by a pf_interval

        >>> from pof.distribution import Distribution
        >>> dist = Distribution(alpha = 50, beta = 50, gamma = 10)
        >>> pf_dist = Distribution.from_pf_interval(dist, 5)
        >>> pf_dist.__dict__
        {'name': 'dist', 'alpha': 50, 'beta': 50, 'gamma': 5}
        """
        alpha = dist.alpha
        beta = dist.beta
        if pf_interval is None:
            gamma = max(0, dist.gamma)
        else:
            gamma = max(0, dist.gamma - pf_interval)

        new_dist = cls(alpha=alpha, beta=beta, gamma=gamma)

        return new_dist

    def cff(self, t_start, t_end):
        t_interval = np.arange(t_start, t_end + 1, 1)

        P_interval = ss.weibull_min.sf(
            t_interval, self.beta, scale=self.alpha, loc=self.gamma
        )

        P = 1 - P_interval / P_interval[0]

        return P

    def update_from_dict(self, dict_data):

        for key, value in dict_data.items():

            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

    def get_value(self, key):
        """
        Takes a key as either a list or a variable name and returns the value stored at that location.

        Usage:
            >>> dist = Distribution(alpha=10, beta = 3, gamma=1)
            >>> dist.get_value(key="alpha")
            10
        """
        if isinstance(key, str):
            value = self.__dict__[key]

        elif isinstance(key, list):
            if len(key) == 1:
                value = self.__dict__[key[0]]
            else:
                value = ()
                for k in key:
                    value = value + (self.__dict__[k],)
        else:
            print("ERROR")

        return value


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
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS, extraglobs={"dist": Distribution()})
    print("Distribution - Ok")
