"""
    Filename: indicator.py
    Description: Contains the code for implementing an indicator class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import numpy as np
import collections
import scipy.stats as ss
from matplotlib import pyplot as plt


# TODO overload methods to avoid if statements and improve speed




class Indicator():

    """

    Methods

        from_dict()

        load_asset_data()

        sim_timeline()

        sim_failure_timeline()

    """

    def __init__(self, parent=None, name='indicator', **kwargs):

        #TODO fix kwargs

        self.name = name

        # Link to parent component for 
        self.component = parent #TODO change this to asset data
        #todo confirm need for parent


    @classmethod
    def load(cls, details=None):
        try:
            ind = cls.from_dict(details)
        except:
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
        
        #Overloaded 
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


        for name, cond in self._timeline.items():
            # Plot with matplotlib
            plt.plot(profile, label=name)

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
            plt.plot(self.t_condition, self.current(), "rd")

class ConditionIndicator(Indicator):

    def __init__(self, parent=None, name='ConditionIndiator',
        pf_curve = 'linear', pf_interval =10, pf_std = 0, 
        threshold_detection=None, threshold_failure=None,
        perfect=100, failed=0, 
        *args, **kwargs
    ):
        super().__init__(name=name, *args, *kwargs)
        # Condition Loss by causes

        # Condition details
        self.pf_curve = None
        self.pf_interval = pf_interval # Default pf_interval when a ConditionIndicator can't have multiple pf_intervals
        self.pf_std = pf_std
        self.pf_curve_params = NotImplemented #TODO for complex condition types
        self.set_pf_curve(pf_curve)

        # Condition
        self.perfect = None
        self.failed = None
        self.decreasing = None
        self.set_limits(perfect = perfect, failed = failed)

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


    def sim_timeline(self, t_end, t_start=0, pf_interval=None, pf_std=None, name=None):
        """
        Returns the timeline that considers all the accumulated degradation
        """
        
        # Set the condition profile if it hasn't been created already
        if pf_interval not in self._profile or pf_std is not None:
            self._set_profile(pf_interval=pf_interval)
  
        self._timeline = self._get_timeline(t_start=t_start, t_stop=t_end, pf_interval=pf_interval)

        return self._timeline


    def _set_profile(self, perfect=None, failed=None, pf_interval=None, pf_std=None, name=None):

        #TODO Illyse - add the other profile types
        """
        Linear: μ(t) = b + a × t
        Exponential: μ(t) = b × exp(a × t)
        Power: μ(t) = b × t a
        Logarithm: μ(t) = a × ln(t) + b
        Lloyd-Lipow: μ(t) = a − (b/t)
        """
        
        # Use the condition parameters if unique parameters aren't provided TODO maybe remove/
        if perfect is None:
            perfect=self.perfect
        
        if failed is None:
            failed = self.failed

        if pf_interval is None:
            pf_interval = self.pf_interval

        if pf_std is None:
            pf_std = self.pf_std

        # Adjust the pf_interval based on the expected variance in pf_std
        if pf_std is not None:
            pf_interval = int(pf_interval + ss.norm.rvs(loc=0, scale=pf_std))

        # Get the time to be investitaged
        x = np.linspace(0, pf_interval, pf_interval + 1)

        if self.pf_curve == 'linear':
    
            m = (failed - perfect) / pf_interval
            b = perfect
            y = m*x + b
        
        elif self.pf_curve == 'step':
            NotImplemented

        elif self.pf_curve == 'exponential' or self.pf_curve == 'exp':
            NotImplemented
        
        self._profile[name] = y
        


    def _get_timeline(self, t_start, t_stop, pf_interval, name=None):  # TODO this probably needs a delay?
        """
        Returns the timeli
        """

        # Validate times
        t_max = len(self._profile[name])
        if t_stop == None:
            t_stop = t_max

        if t_start > t_stop:
            t_start = t_stop

        if t_stop < 0:
            t_start = t_start - t_stop
            t_stop = 0

        cp = self._profile[name][min(t_start, t_max):min(t_stop, t_max) + 1]

        # Adjust for the accumulated condition
        cp = cp - self.get_accumulated()
        cp[cp < self.failed] = 0

        # Fill the start with the current condtiion
        if t_start < 0:
            cp = np.append(np.full(t_start * -1, cp[0]), cp)

        # Fill the end with the failed condition
        n_after_failure = t_stop - t_start - len(cp) + 1
        if n_after_failure > 0:
            cp = np.append(cp, np.full(max(0, n_after_failure), self.failed))

        return cp


    def sim_failure_timeline(self):
        
        #Overloaded 
        NotImplemented


    def restore(self):

        NotImplemented

    # ********************* Get Methods **********************

    def get_accumulated(self, name=None): #TODO make this work for arrays of names

        if name is None:
            # Get all the total acumulated condition
            accumulated = sum(self._accumulated.values())
        
        else:
            
            # Get the accumulated condition for a single name
            if isinstance(name, str):
                if name in self._accumulated:
                    accumulated = self._accumulated[name]

            # Get the accumulated condition for a list of names
            elif isinstance(ob, collections.Iterable):
                accumulated = sum([self._accumulated.get(key, 0) for key in name])
            else:
                raise TypeError("name should be a string or iterable")

        return accumulated

        

    # ********************* Set Methods **********************

    def _set_accumulated(self, accumulated, name=None):
        
        if name is None:
            name =self.name

        self._accumulated[name] = accumulated

    def set_condition(self, condition, name=None):

        if name is None:
            name =self.name

        self._accumlated[name] = condition


    def set_pf_curve(self, pf_curve):

        valid_pf_curve = ['linear', 'step', 'linear-legacy']
        if pf_curve in valid_pf_curve:
            self.pf_curve = pf_curve
        else:
            raise ValueError("pf_curve must be from: %s" %(valid_pf_curve))

    def set_threshold(self, detection=None, failure = None):
        if detection is None:
            self.threshold_detection = self.perfect
        else:
            self.threshold_detection = detection

        if failure is None:
            self.threshold_failure = self.failed
        else:
            self.threshold_failure = failure

    def set_limits(self, perfect, failed):
        #TODO make sure these tests work for bool and int

        # Set perfect
        if perfect > failed:
            self.decreasing = True
            self.perfect = perfect
            self.failed = failed
        else:
            self.decreasing = False
            self.perfect = failed
            self.failed = perfect


class PoleSafetyFactor(Indicator):

    def __init__(self, component, failed = 1, decreasing=True):
        super().__init__(self)

        self.component = component

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
        

    def safety_factor(self, method = 'simple'):

        if method == 'simple':
            sf = self._safety_factor(
                agd = self.component.indicator['external_diameter'].perfect,
                czd = self.component.indicator['external_diameter'].get_condition_profile(0,200),
                wt = self.component.indicator['wall_thickness'].get_condition_profile(0,200),
                margin = 4
            )

        elif method == 'actual':
            sf = self._safety_factor(
                agd = self.component.conditions['external_diameter'].perfect,
                czd = self.component.conditions['external_diameter'],
                wt = self.component.conditions['wall_thickness'],
                pole_load = self.component.info['pole_load'],
                pole_strength = self.component.info['pole_strength'],
            )

        return sf


    def _safety_factor(self, agd, czd, wt, pole_strength=None, pole_load=None, margin=4):
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

        sf = margin * (czd**4 - (czd - 2*wt)**4) / (agd**3 * czd)

        return sf



"""
ConditionIndicator
SafetyFactorIndicator

"""