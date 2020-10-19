"""
    Filename: indicator.py
    Description: Contains the code for implementing an indicator class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

from dataclasses import dataclass, field
from typing import Dict
import collections
import copy
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib import pyplot as plt

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.load import Load
from pof.helper import str_to_dict
from config import config

cf = config["Indicator"]


# TODO move timeline to indicator
# TODO overload methods to avoid if statements and improve speed
# TODO make sure everything works for conditions in both direction
# TODO robust testing
# TODO move threshold down into condition indciator so indicator is purely bool


@dataclass
class Indicator(Load):

    """

    Methods

        from_dict()

        load_asset_data()

        sim_timeline()

        sim_failure_timeline()

    """

    PF_CURVES = ["linear", "step"]

    name: str = "indicator"
    pf_curve: str = "step"
    pf_interval: int = 0  # TODO change pf_interval to None?
    pf_std: int = 0
    perfect: bool = False
    failed: bool = True
    decreasing: bool = field(init=False)

    threshold_detection: int = None
    threshold_failure: int = None

    initial: int = None

    _profile: Dict = field(init=False, repr=False)
    _timeline: Dict = field(init=False, repr=False)
    _timelines: Dict = field(init=False, repr=False)

    def __post_init__(self):

        self.set_limits(perfect=self.perfect, failed=self.failed)
        self.set_threshold(
            detection=self.threshold_detection, failure=self.threshold_failure
        )
        self.set_initial(initial=self.initial)
        self.set_pf_curve(pf_curve=self.pf_curve)
        self.set_pf_interval(pf_interval=self.pf_interval)
        self.reset()

    @classmethod
    def from_dict(cls, details=None):
        """
        Overloaded factory for creating indicators
        """
        if isinstance(details, dict):
            pf_curve = details.get("pf_curve", None)

            if pf_curve in ["linear", "step"]:

                ind = ConditionIndicator(**details)

            elif pf_curve in ["ssf_calc", "dsf_calc"]:

                ind = PoleSafetyFactor(**details)

            elif pf_curve is None:
                ind = Indicator(**details)

            else:
                raise ValueError("Invalid Indicator Type")
        
        else:
            raise TypeError("Dictionary expected")

        return ind

    def sim_timeline(self, *args):
        logging.debug("Non overloaded function called")
        NotImplemented

    def sim_failure_timeline(self, *args, **kwargs):
        logging.debug("Non overloaded function called")

        NotImplemented

    def restore(self, *args, **kwargs):
        logging.debug("Non overloaded function called")
        NotImplemented

    def reset_any(self, *args, **kwargs):
        logging.debug("Non overloaded function called")
        NotImplemented

    def reset(self):

        self._profile = dict()
        self._timeline = dict()
        self._timelines = dict()

    def reset_for_next_sim(self):
        None

    def reset_to_perfect(self):
        None

    def set_pf_curve(self, pf_curve):
        if pf_curve in self.PF_CURVES:
            self.pf_curve = pf_curve
        else:
            raise ValueError("pf_curve must be from: %s" % (self.PF_CURVES))

    def set_pf_interval(self, pf_interval=None):

        # TODO add robust testing around pf_interval non negative numbers etc
        if pf_interval is None:
            if self.pf_interval is None:
                if cf.getboolean("use_default"):
                    logging.warning(
                        "%s - %s - pf_interval set to DEFAULT %s",
                        self.__class__.__name__,
                        self.name,
                        cf["PF_INTERVAL"],
                    )
                    self.pf_interval = cf["PF_INTERVAL"]
                else:
                    if False:  # TODO revist whether pf_interval is required
                        raise ValueError(
                            "%s - %s - pf_interval required"
                            % (self.__class__.__name__, self.name)
                        )
        else:
            self.pf_interval = pf_interval

    def set_limits(self, perfect=None, failed=None):
        # TODO Add test make sure these tests work for bool and int

        if perfect is None:
            if self.perfect is None:
                if cf.getboolean("use_default"):
                    self.perfect = cf["PERFECT"]
                else:
                    raise ValueError(
                        "%s - %s - perfect value required"
                        % (self.__class__.__name__, self.name)
                    )
        else:
            self.perfect = perfect

        if failed is None:
            if self.failed is None:
                if cf.getboolean("use_default"):
                    self.failed = cf["FAILED"]
                else:
                    raise ValueError(
                        "%s - %s - failed value required"
                        % (self.__class__.__name__, self.name)
                    )
        else:
            self.perfect = perfect

        # Set perfect
        if self.perfect > self.failed:
            self.decreasing = True
        else:
            self.decreasing = False

        self.upper = abs(self.perfect - self.failed)

    def set_initial(self, initial=None):
        """ Set the intial """
        # TODO add checks to make sure it is a valid value

        if initial is None:
            self.initial = self.perfect
        else:
            self.initial = initial

    def set_threshold(self, detection=None, failure=None):
        if detection is None:
            if self.threshold_detection is None:
                self.threshold_detection = self.perfect
            else:
                self.threshold_detection = detection

        if failure is None:
            if self.threshold_failure is None:
                self.threshold_failure = self.failed
            else:
                self.threshold_failure = failure

    # ****************** Get methods **************

    # TODO split out into conditionIndicator and simplify this method for Indicator to make it faster

    def agg_timeline(self):

        if self.decreasing:
            timeline = self.perfect - (
                self.perfect - np.array(list(self._timeline.values()))
            ).sum(axis=0)
            timeline[timeline < self.failed] = self.failed
        else:
            timeline = self.perfect + (
                np.array(list(self._timeline.values())) - self.perfect
            ).sum(axis=0)
            timeline[timeline > self.failed] = self.failed
        return timeline

    def agg_timelines(self):
        """
        Takes a dictionary of timelines and returns the aggregated timelines
        """
        agg_timeline = []
        for timeline in self._timelines.values():
            if self.decreasing:
                etl = self.perfect - (
                    self.perfect - np.array(list(timeline.values()))
                ).sum(axis=0)
                etl[etl < self.failed] = self.failed
            else:
                etl = self.perfect + (
                    np.array(list(timeline.values())) - self.perfect
                ).sum(axis=0)
                etl[etl > self.failed] = self.failed

            agg_timeline.append(etl)

        return np.stack(agg_timeline)

    def get_timeline(self, name=None):
        # maybe add t_start?
        """ Returns the timeline for a name if it is in the key or if no key is passed and None is not a key, it aggregates all timelines"""
        try:
            timeline = self._timeline[name]
        except KeyError as name_not_in_timeline:
            if name is None:
                timeline = self.agg_timeline()
            else:
                raise KeyError(
                    "Name - %s - is not in %s %s timeline"
                    % (name, self.__class__.__name__, self.name)
                ) from name_not_in_timeline

        return timeline

    #  ********************* Interface methods ***********************

    def plot_profile(self):

        for name, profile in self._profile.items():
            plt.plot(profile, label=name)

        plt.title("Indicator Profile")
        plt.show()

    def plot_timeline(self):

        for name, timeline in self._timeline.items():
            # Plot with matplotlib
            plt.plot(timeline, label=name)

        plt.title("Indicator Timeline")
        plt.show()

    def plot_timelines(self, i=None, n=None):

        if i is not None:
            if n is None:
                self._plot_timeline(self._timelines[i])
            else:
                for idx in range(i, n):
                    plt.plot(self._timeline[idx])

        plt.show()

    def _plot_timeline(self, _timeline=None):
        if _timeline is None:
            _timeline = self._timeline

        for cause, timeline in _timeline.items():
            plt.plot(timeline)
            # plt.plot(self.t_condition, self.current(), "rd")

    def update_from_dict(self, dict_data):

        for key, value in dict_data.items():

            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                raise KeyError(
                    'ERROR: Cannot update "%s" from dict with key %s'
                    % (self.__class__.__name__, key)
                )

    """def expected_condition(self, conf=0.5):
        ec = self.get_timeline()
        return self._expected_condition(ec, conf)"""

    def _expected_condition(self, ec, conf):
        """
        Returns the expected condition based
        """
        # TODO make work for all condition levels loss:bool=False

        mean = ec.mean(axis=0)
        sigma = ec.std(axis=0)

        # TODO maybe add np.sqr(len(ec)) to make it stderr

        # Calculate bounds
        upper = ss.norm.ppf((1 - (1 - conf) / 2), loc=mean, scale=sigma)
        lower = ss.norm.ppf(((1 - conf) / 2), loc=mean, scale=sigma)

        # Adjust upper and lower to the mean if there is not variance
        no_variance = pd.isna(sigma) | (sigma == 0)
        upper[no_variance] = mean[no_variance]
        lower[no_variance] = mean[no_variance]

        if self.decreasing:
            upper[upper > self.perfect] = self.perfect
            lower[lower < self.failed] = self.failed
        else:
            upper[upper > self.failed] = self.failed
            lower[lower < self.perfect] = self.perfect

        expected = dict(
            lower=lower,
            mean=mean,
            upper=upper,
        )

        return expected

    def save_timeline(self, idx=None):
        self._timelines[idx] = copy.deepcopy(self._timeline)


@dataclass
class ConditionIndicator(Indicator):

    # Class Variables
    PF_CURVES = ["linear", "step"]

    name: str = "ConditionIndicator"

    def __post_init__(self):
        super().__post_init__()

        self.pf_curve_params = NotImplemented  # TODO for complex condition types

        # Current accumulation
        self._accumulated = dict()
        self._set_accumulated(
            accumulated=abs(self.perfect - self.initial), name="initial"
        )

    # ********************** Timeline methods ******************************

    def mc_timeline(self, t_end, t_start=0, n_iterations=100):
        for i in range(n_iterations):
            self.sim_timeline(t_start=t_start, t_stop=t_end)
            self.save_timeline()
            self.reset_for_next_sim()

    def sim_timeline(
        self,
        t_stop=None,
        t_delay=0,
        t_start=0,
        pf_interval=None,
        pf_std=0,
        name=None,
    ):
        """
        Returns the timeline that considers all the accumulated degradation and save timeline
        """
        # TODO make the t_delay more elegeant and remove duplication from failure_mode

        if name not in self._timeline:
            self._timeline[name] = self._sim_timeline(
                t_start=t_start, t_stop=t_stop, pf_interval=pf_interval, name=name
            )

        else:
            if t_delay is None:
                t_delay = 0

            self._timeline[name][t_delay:] = self._sim_timeline(
                t_start=t_start,
                t_stop=t_stop,
                pf_interval=pf_interval,
                pf_std=pf_std,
                name=name,
            )

        return self._timeline[name][t_delay:]

    def _sim_timeline(
        self, t_stop=None, t_start=0, pf_interval=None, pf_std=None, name=None
    ):
        """
        Returns the timeline that considers all the accumulated degradation
        """

        # Use the condition parameters if unique parameters aren't provided TODO maybe remove
        if pf_interval is None:
            pf_interval = self.pf_interval

        if pf_std is None:
            pf_std = self.pf_std

        # Adjust the pf_interval based on the expected variance in pf_std
        if pf_std is not None:
            pf_interval = int(pf_interval + ss.norm.rvs(loc=0, scale=pf_std))

        # Set the condition profile if it hasn't been created already or if uncertainty is needed
        if pf_interval not in self._profile:
            self._set_profile(pf_interval=pf_interval, name=name)

        # Get the timeline
        timeline = self._acc_timeline(
            t_start=t_start, t_stop=t_stop, pf_interval=pf_interval, name=name
        )

        return timeline

    def _set_profile(
        self, perfect=None, failed=None, pf_interval=None, pf_std=None, name=None
    ):

        # TODO Illyse - add the other profile types
        # TODO maybe make this work using pf_interval and name so that it doesn't do as much recalcuting
        """
        Linear: μ(t) = b + a × t
        Exponential: μ(t) = b × exp(a × t)
        Power: μ(t) = b × t a
        Logarithm: μ(t) = a × ln(t) + b
        Lloyd-Lipow: μ(t) = a − (b/t)
        """

        # Use the condition parameters if unique parameters aren't provided TODO maybe remove/
        if perfect is None:
            perfect = self.perfect

        if failed is None:
            failed = self.failed

        if pf_interval is None:
            pf_interval = self.pf_interval

        if pf_std is None:
            pf_std = self.pf_std

        # Get the time to be investitaged
        x = np.linspace(0, pf_interval, pf_interval + 1)

        if self.pf_curve == "linear":
            # Prevent zero division error
            if pf_interval <= 0:
                m = 0
            else:
                m = (failed - perfect) / pf_interval
            b = perfect
            y = m * x + b

        elif self.pf_curve == "step":
            if pf_interval == 0:
                y = np.full(1, self.failed)
            else:
                y = np.full(pf_interval, self.perfect)

        elif self.pf_curve == "exponential" or self.pf_curve == "exp":
            NotImplemented

        self._profile[pf_interval] = y

    def _acc_timeline(self, t_start=0, t_stop=None, pf_interval=None, name=None):
        # TODO this probably needs a delay?
        """
        Returns the timeli
        """

        # Validate times
        t_max = len(self._profile[pf_interval]) - 1
        if t_stop == None:
            t_stop = t_max

        if t_start > t_stop:
            t_start = t_stop

        if t_stop < 0:
            t_start = t_start - t_stop
            t_stop = 0

        profile = self._profile[pf_interval][
            max(0, min(t_start, t_max)) : min(t_stop, t_max) + 1
        ]

        # Adjust for the accumulated condition
        accumulated = self.get_accumulated(name=name)
        if accumulated > 0:
            profile = profile - accumulated
            profile[profile < self.failed] = self.failed

        # Fill the start with the current condtiion
        if t_start < 0:
            profile = np.append(np.full(t_start * -1, profile[0]), profile)

        # Fill the end with the failed condition
        n_after_failure = t_stop - t_start - len(profile) + 1
        if n_after_failure > 0:
            profile = np.append(profile, np.full(max(0, n_after_failure), self.failed))

        return profile

    def sim_failure_timeline(
        self,
        t_stop=None,
        t_delay=0,
        t_start=0,
        pf_interval=None,
        pf_std=None,
        name=None,
    ):
        # TODO this probably needs a delay? and can combine with condtion profile to make it simpler
        """
        Return a sample failure schedule for the condition
        """

        if self.decreasing == True:
            tl_f = self._timeline[name][t_delay:] <= self.threshold_failure
        else:
            tl_f = self._timeline[name][t_delay:] >= self.threshold_failure

        return tl_f

    # ************** Simulate Condition ***************

    def sim(self, t, name=None):
        """
        Return the condition at time t
        """
        self.sim_timeline(t_stop=t, name=name)

        return self._timeline[name][t]

    # ********************* Get Methods **********************

    def get_condition(self):
        if self.decreasing:
            return self.perfect - self.get_accumulated()
        else:
            return self.perfect + self.get_accumulated()

    def get_timeline(self, name=None):
        return self._timeline[name]

    def get_timelines(self):
        return self._timelines

    def get_accumulated(self, name=None):  # TODO make this work for arrays of names

        if name is None:
            # Get all the total acumulated condition
            accumulated = sum(self._accumulated.values())

        else:

            # Get the accumulated condition for a single name
            if isinstance(name, str):
                accumulated = self._accumulated.get(name, 0) + self._accumulated.get(
                    "permanent", 0
                )

            # Get the accumulated condition for a list of names
            elif isinstance(name, collections.Iterable):
                accumulated = sum(
                    [self._accumulated.get(key, 0) for key in name]
                ) + self._accumulated.get("permanent", 0)
            else:
                raise TypeError("name should be a string or iterable")

        return accumulated

    # ********************* Set Methods **********************

    def set_t_condition(self, t, name=None):
        """ Set the condition base on a time t"""
        condition = float(self._timeline[name][t])
        self.set_condition(condition)

    def set_condition(self, condition, name=None):
        """ Set the condition based on a condition"""
        # TODO consider impact of other impacts

        if self.decreasing:
            accumulated = min(
                max(0, self.perfect - condition), self.perfect - self.failed
            )
        else:
            accumulated = min(
                max(0, condition - self.perfect), self.failed - self.perfect
            )

        self._set_accumulated(accumulated=accumulated, name=name)

    def _set_accumulated(self, accumulated, name=None):
        """
        Add the accumulated degradation checking that it won't exceed the limits given intial condition
        """
        # check accumulated will not exceed the maximum allowable condition
        current = (
            self.get_accumulated()
        )  # TODO is this needed - self.get_accumulated(name=name)

        # max accumulation - current - initial
        self._accumulated[name] = min(
            accumulated,
            abs(self.perfect - self.failed) - current,
        )

    def reset(self):

        super().reset()
        self._reset_accumulated()

    def reset_for_next_sim(self):
        self._reset_accumulated(
            accumulated=abs(self.perfect - self.initial), name="initial"
        )

    def reset_to_perfect(self):
        self._reset_accumulated()

    def reset_any(self, target=0, method="reset", axis="time", permanent=False):
        """
        # TODO make this work for all the renewal processes (as-bad-as-old, as-good-as-new, better-than-old, grp)
        """

        # Error with time reset, different method required.

        if method == "reduction_factor":
            # accumulated = (abs(self.perfect - self.get_accumulated())) * (1 - target)
            accumulated = self.get_accumulated() * (1 - target)

        elif method == "reverse":

            accumulated = self.get_accumulated() - target

        elif method == "set":
            accumulated = target

        # Calculate the accumulated condition TODO not working
        if axis == "time":

            NotImplemented

        elif axis == "condition":

            """if self.decreasing:
                accumulated = min(max(self.failed, accumulated), self.perfect)
            else:
                accumulated = max(min(self.failed, accumulated), self.perfect)"""

            self._reset_accumulated(accumulated, permanent=permanent)

    def _reset_accumulated(self, accumulated=0, name=None, permanent=False):
        """
        Reset the accumulated condition with an option to add permanent condition loss
        """

        # Maintain permanent condition loss if set
        if permanent:
            accumulated = accumulated + self._accumulated.get("permanent", 0)
            name = "permanent"

        self._accumulated = dict()
        self._set_accumulated(name=name, accumulated=accumulated)

    def update_from_dict(self, keys):
        """
        Takes a dict of key values and updates an attribute in indicator
        """
        for key, value in keys.items():

            try:
                super().update_from_dict({key: value})
            except KeyError:
                if key in self.__dict__:
                    self.__dict__[key] = value
                else:
                    raise KeyError(
                        'ERROR: Cannot update "%s" - %s from dict with key %s'
                        % (self.__class__.__name__, self.name, key)
                    )

    def expected_condition(self, conf=0.5):
        ec = self.agg_timelines()
        return self._expected_condition(ec, conf)


# TODO overload get method so the None key isnt' needed for _timeline


@dataclass
class PoleSafetyFactor(Indicator):

    # Class Variables
    PF_CURVES = ["ssf_calc", "dsf_calc"]

    failed: int = 1
    decreasing: int = True
    component = None

    def set_t_condition(self, *args, **kwargs):
        """No actions required"""
        None

    def link_component(self, component):
        self.component = component

    def sim_timeline(self, t_delay=0, *args, **kwargs):
        """
        Overload safety factor
        """
        self._timeline[None] = self.safety_factor("simple")
        return self._timeline[None][t_delay:]

    def sim_failure_timeline(self, t_delay=0, *args, **kwargs):

        if self.decreasing == True:
            tl_f = self._timeline[None][t_delay:] <= self.threshold_failure
        else:
            tl_f = self._timeline[None][t_delay:] >= self.threshold_failure

        return tl_f

    def safety_factor(self, method="simple"):

        if method == "simple":
            sf = self._safety_factor(
                agd=self.component.indicator["external_diameter"].perfect,
                czd=self.component.indicator["external_diameter"].get_timeline(),
                wt=self.component.indicator["wall_thickness"].get_timeline(),
                margin=4,
            )

        elif method == "actual":
            sf = self._safety_factor(
                agd=self.component.indicator["external_diameter"].perfect,
                czd=self.component.indicator["external_diameter"],
                wt=self.component.indicator["wall_thickness"],
                pole_load=self.component.info["pole_load"],
                pole_strength=self.component.info["pole_strength"],
            )

        return sf

    def _safety_factor(
        self, agd, czd, wt, pole_strength=None, pole_load=None, margin=4
    ):
        """
        Calculates the safety factor using a margin of 4 if the pole load and pole strength are not available

            Params:
                agd:    above ground diamater

                czd:    critical zone diameter

                wt:     wall thickness

                margin: the safety margin used when designing the pole

        """

        if pole_load is not None and pole_strength is not None:

            margin = pole_strength / pole_load

        # Supress divide by zero error and fill na with zero
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = margin * (czd ** 4 - (czd - 2 * wt) ** 4) / (agd ** 3 * czd)
            sf[np.isnan(sf)] = 0

        return sf

    def update_from_dict(self, keys):

        for key, value in keys.items():

            try:
                super().update_from_dict({key: value})
            except KeyError:
                if key in self.__dict__:
                    self.__dict__[key] = value
                else:
                    raise KeyError(
                        'ERROR: Cannot update "%s" - %s from dict with key %s'
                        % (self.__class__.__name__, self.name, key)
                    )

    def expected_condition(self, conf=0.5):
        ec = self.agg_timelines()
        return self._expected_condition(ec, conf)


# class ActualSafetyFactor(PoleSafetyFactor):


if __name__ == "__main__":
    indicator = Indicator()
    print("Indicator - Ok")
