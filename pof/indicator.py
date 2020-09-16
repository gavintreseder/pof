"""
    Filename: indicator.py
    Description: Contains the code for implementing an indicator class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import numpy as np
import collections
import scipy.stats as ss
from matplotlib import pyplot as plt
from enum import Enum


# TODO overload methods to avoid if statements and improve speed
# TODO make sure everything works for conditions in both direction
# TODO robust testing


class IndicatorValid(Enum):
    PF_CURVE = ["linear", "step", "linear-legacy"]


class Indicator:

    """

    Methods

        from_dict()

        load_asset_data()

        sim_timeline()

        sim_failure_timeline()

    """

    def __init__(self, parent=None, name="indicator", pf_curve="linear", **kwargs):

        # TODO fix kwargs

        self.name = name

        # Link to parent component for
        self.component = parent  # TODO change this to asset data
        # todo confirm need for parent

        self.pf_curve = pf_curve

    @classmethod
    def load(cls, details=None):
        try:
            ind = cls.from_dict(details)
        except Exception:
            ind = cls()
            print("Error loading Indicator data")
        return ind

    @classmethod
    def from_dict(cls, details=None):
        try:
            ind = cls(**details)
        except:
            ind = cls()
            print("Error loading Indicator data from dictionary")
        return ind

    def link(self, **args):

        NotImplemented

    def sim_indicator_timeline(self):

        # Overloaded
        NotImplemented

    def sim_failure_timeline(self):

        # Overloaded
        NotImplemented

    def restore(self):

        NotImplemented

    def reset(self):

        self._profile = dict()
        self._timeline = dict()
        self._timelines = dict()
        NotImplemented

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


class ConditionIndicator(Indicator):
    def __init__(
        self,
        parent=None,
        name="ConditionIndiator",
        pf_curve="linear",
        pf_interval=10,
        pf_std=0,
        threshold_detection=None,
        threshold_failure=None,
        perfect=100,
        failed=0,
        *args,
        **kwargs
    ):
        super().__init__(name=name, *args, *kwargs)
        # Condition Loss by causes

        # Condition details
        self.pf_curve = pf_curve
        self.pf_interval = pf_interval  # Default pf_interval when a ConditionIndicator can't have multiple pf_intervals
        self.pf_std = pf_std
        self.pf_curve_params = NotImplemented  # TODO for complex condition types
        self.set_pf_curve(pf_curve)

        # Condition
        self.perfect = None
        self.failed = None
        self.decreasing = None
        self.set_limits(perfect=perfect, failed=failed)

        # Detection and failure thresholds
        self.threshold_detection = None
        self.threshold_failure = None
        self.set_threshold(detection=threshold_detection, failure=threshold_failure)

        # Current accumulation
        self._accumulated = dict()

        # Profile and timeslines
        self._profile = dict()
        self._timeline = dict()
        self._timelines = dict()

    # ********************** Timeline methods ******************************

    def sim_timeline(
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

        self._timeline[name] = self._get_timeline(
            t_start=t_start, t_stop=t_stop, pf_interval=pf_interval, name=name
        )

        return self._timeline[name]

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

            m = (failed - perfect) / pf_interval
            b = perfect
            y = m * x + b

        elif self.pf_curve == "step":
            y = np.full(self.pf_interval, self.perfect)

        elif self.pf_curve == "exponential" or self.pf_curve == "exp":
            NotImplemented

        self._profile[pf_interval] = y

    def _get_timeline(
        self, t_start, t_stop=None, pf_interval=None, name=None
    ):  # TODO this probably needs a delay?
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

        cp = self._profile[pf_interval][
            max(0, min(t_start, t_max)) : min(t_stop, t_max) + 1
        ]

        # Adjust for the accumulated condition
        accumulated = self.get_accumulated()
        if accumulated > 0:
            cp = cp - self.get_accumulated()
            cp[cp < self.failed] = self.failed

        # Fill the start with the current condtiion
        if t_start < 0:
            cp = np.append(np.full(t_start * -1, cp[0]), cp)

        # Fill the end with the failed condition
        n_after_failure = t_stop - t_start - len(cp) + 1
        if n_after_failure > 0:
            cp = np.append(cp, np.full(max(0, n_after_failure), self.failed))

        return cp

    def sim_failure_timeline(
        self, t_stop=None, t_start=0, pf_interval=None, pf_std=None, name=None
    ):  # TODO this probably needs a delay? and can combine with condtion profile to make it simpler
        """
        Return a sample failure schedule for the condition
        """

        cp = self.sim_timeline(
            t_stop=t_stop,
            t_start=t_start,
            pf_interval=pf_interval,
            pf_std=pf_std,
            name=name,
        )

        if self.decreasing == True:
            tl_f = cp <= self.threshold_failure
        else:
            tl_f = cp >= self.threshold_failure

        return tl_f

    def restore(self):

        NotImplemented

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

    def get_accumulated(self, name=None):  # TODO make this work for arrays of names

        if name is None:
            # Get all the total acumulated condition
            accumulated = sum(self._accumulated.values())

        else:

            # Get the accumulated condition for a single name
            if isinstance(name, str):
                if name in self._accumulated:
                    accumulated = self._accumulated[name]

            # Get the accumulated condition for a list of names
            elif isinstance(name, collections.Iterable):
                accumulated = sum([self._accumulated.get(key, 0) for key in name])
            else:
                raise TypeError("name should be a string or iterable")

        return accumulated

    # ********************* Set Methods **********************

    def _set_accumulated(self, accumulated, name=None):

        if name is None:
            name = self.name

        # check accumulated will not exceed the maximum allowable condition
        current = self.get_accumulated()

        self._accumulated[name] = min(
            accumulated, abs(self.perfect - self.failed) - current - accumulated
        )

    def set_condition(self, condition, name=None):
        # TODO consider impact of other impacts

        if name is None:
            name = self.name

        if self.decreasing:
            self._accumulated[name] = min(
                max(0, self.perfect - condition), self.perfect - self.failed
            )
        else:
            self._accumulated[name] = min(
                max(0, condition - self.perfect), self.failed - self.perfect
            )

    def set_pf_curve(self, pf_curve):

        if pf_curve in IndicatorValid.PF_CURVE.value:
            self.pf_curve = pf_curve
        else:
            raise ValueError("pf_curve must be from: %s" % (IndicatorValid.PF_CURVE))

    def set_threshold(self, detection=None, failure=None):
        if detection is None:
            self.threshold_detection = self.perfect
        else:
            self.threshold_detection = detection

        if failure is None:
            self.threshold_failure = self.failed
        else:
            self.threshold_failure = failure

    def set_limits(self, perfect, failed):
        # TODO make sure these tests work for bool and int

        # Set perfect
        if perfect > failed:
            self.decreasing = True

        else:
            self.decreasing = False

        self.perfect = perfect
        self.failed = failed

    def reset(self):

        super().reset()
        self._reset_accumulated()

    def reset_any(self, target=0, method="reset", axis="time", permanent=False):
        """
        # TODO make this work for all the renewal processes (as-bad-as-old, as-good-as-new, better-than-old, grp)
        """

        # Error with time reset, different method required.

        if method == "reduction_factor":
            accumulated = (abs(self.perfect - self.get_accumulated())) * (1 - target)

        elif method == "reverse":

            accumulated = self.get_accumulated() - target

        elif method == "set":
            accumulated = target

        # Calculate the accumulated condition TODO not working
        if axis == "time":

            NotImplemented

        elif axis == "condition":

            if self.decreasing:
                accumulated = min(max(self.failed, accumulated), self.perfect)
            else:
                accumulated = max(min(self.failed, accumulated), self.perfect)

            self._reset_accumulated(accumulated, permanent=permanent)

    def _reset_accumulated(self, accumulated=0, name=None, permanent=False):

        # Maintain permanent condition loss if set
        if permanent:
            existing_permanent = self._accumulated.get("permanent", 0)
            accumulated = permanent + existing_permanent
            name = "permanent"

        self._accumulated = dict()
        self._set_accumulated(name=name, accumulated=accumulated)


class PoleSafetyFactor(Indicator):
    def __init__(self, failed=1, decreasing=True, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        # Condition detection and limits
        self.decreasing = decreasing
        self.threshold_failure = failed

    def sim_failure_timeline(self):
        """
        Determine if the indicator hsa failed
        """
        # Get the timeline
        timeline = self.safety_factor()

        # Check
        if self.decreasing == True:
            timeline = timeline <= self.threshold_failure
        else:
            timeline = timeline >= self.threshold_failure

        return timeline

    def safety_factor(self, method="simple"):

        if method == "simple":
            sf = self._safety_factor(
                agd=self.component.indicator["external_diameter"].perfect,
                czd=self.component.indicator["external_diameter"].get_condition_profile(
                    0, 200
                ),
                wt=self.component.indicator["wall_thickness"].get_condition_profile(
                    0, 200
                ),
                margin=4,
            )

        elif method == "actual":
            sf = self._safety_factor(
                agd=self.component.conditions["external_diameter"].perfect,
                czd=self.component.conditions["external_diameter"],
                wt=self.component.conditions["wall_thickness"],
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

        sf = margin * (czd ** 4 - (czd - 2 * wt) ** 4) / (agd ** 3 * czd)

        return sf


"""
ConditionIndicator
SafetyFactorIndicator

"""

if __name__ == "__main__":
    indicator = Indicator()
    print("Indicator - Ok")