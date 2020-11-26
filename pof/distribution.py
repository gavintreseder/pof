import copy
from dataclasses import dataclass, field
import logging

import numpy as np
import scipy.stats as ss

# Change the system path when this is run directly
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.load import Load
from config import config
from pof.decorators import check_arg_positive, coerce_arg_type, check_arg_type

cf = config["Distribution"]

# TODO Extend so that it works for all the common distributions

# class Manager(object):

#     """

#     #TODO Maybe move the set and update methods on the manager objects?

#     """
#     def set_obj(self, attr, d_type, value):
#         """

#         value = {'tasks':{'inspection':{'t_interval':10}}}
#         """

#         try:
#             if value is None:
#                 setattr(self, attr, dict())

#             # Add the value to the dictionary if it is an object of that type
#             elif isinstance(value, d_type):
#                 getattr(self, attr)[value.name] = value

#             # Check if the input is an iterable
#             elif isinstance(value, Iterable):

#                 # Create an object from the dict
#                 if all([hasattr(d_type, attr) for attr in value]):
#                     new_object = d_type.from_dict(value)
#                     getattr(self, attr)[new_object.name] = new_object

#                 # Create an object from the dict of dict/objects
#                 else:
#                     for key, val in value.items():

#                         if isinstance(val, d_type):
#                             getattr(self, attr)[val.name] = val

#                         else:
#                             new_object = d_type.from_dict(val)
#                             getattr(self, attr)[new_object.name] = new_object

#             else:
#                 raise ValueError


from collections.abc import MutableMapping


class PofContainer(MutableMapping):
    """A dictionary that changes the key if the name of the pof object it is storing changes"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def update_from_dict(self, data):

        for key, val in data.items():

            # Update with the
            self.store[key].update_from_dict(val)

            # Check if the name has been updated
            if key != self.store[key].name:

                logging.debug("Updating key to match name change")
                new_key = self.store[key].name

                # Change the key if it is already in the dict
                if new_key in self.store:
                    new_key = str(new_key).join(".1")
                    self.store[key].name = new_key
                    logging.debug(
                        "Key %s is already in use. Name and key changed to %s",
                        key,
                        new_key,
                    )

                # Update the key
                self.store[new_key] = self.store[key]
                del self.store[key]


class DistributionManager(PofContainer):

    pf_interval = 0

    def __setitem__(self, item, value):

        # value = set_obj
        # item = value.name
        untreated = copy.copy(self.get("untreated", None))
        super(DistributionManager, self).__setitem__(item, value)

        if untreated != self.get("untreated", None):
            init = Distribution.from_pf_interval(
                self.get("untreated"), self.pf_interval
            )
            init.name = "init"
            self["init"] = init


class Distribution(Load):

    """
    Usage:

    First import Distribution using the distribution package

        >>> from distribution import Distribution

    Create a Distribution object

        >>> Distribution()
        Distribution(name=...)

    """

    # name: str = field(default_factory=lambda: cf.get("name"))
    # alpha: int = field(default_factory=lambda: cf.get("alpha"))
    # beta: int = field(default_factory=lambda: cf.get("beta"))
    # gamma: int = field(default_factory=lambda: cf.get("gamma"))

    # Class Variables
    TIME_VARIABLES = ["pf_interval"]
    
    def __init__(self, name="dist", alpha=50, beta=1.5, gamma=10, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    @coerce_arg_type
    @check_arg_positive("value")
    def alpha(self, value: float):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    @coerce_arg_type
    @check_arg_positive("value")
    def beta(self, value: float):
        self._beta = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    @coerce_arg_type
    @check_arg_positive("value")
    def gamma(self, value: float):
        self._gamma = value

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

    @classmethod
    def demo(cls, scenario=None):

        if scenario is None:
            data = dict(
                name="slow_aging",
                alpha=100,
                beta=2,
                gamma=10,
            )

        return cls.from_dict(data)

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

    # def update_from_dict(self, dict_data):

    #     for key, value in dict_data.items():

    #         if key in self.__dict__:
    #             self.__dict__[key] = value
    #         else:
    #             print('ERROR: Cannot update "%s" from dict' % (self.__class__.__name__))

    # def almost_equal(self, other, decimal=1):


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
