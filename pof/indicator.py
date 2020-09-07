"""
    Filename: indicator.py
    Description: Contains the code for implementing an indicator class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

# TODO overload methods to avoid if statements and improve speed

class Indicator():

    """

    Methods

        load_asset_data()

        sim_timeline()

        sim_failure_timeline()

    """

    def __init__(self, parent, failed = 1, name='indicator', **kwargs):

        #TODO fix kwargs

        # Link to parent component for 
        self.component = parent #TODO change this to asset data

        # Condition detection and limits
        self.decreasing = True
        self.threshold_failure = failed

    def link(self, **args):
        

        NotImplemented


    def sim_indicator_timeline(self):

        # Overloaded
        NotImplemented

    def sim_failure_timeline(self):
        
        #Overloaded 
        NotImplemented


class ConditionIndicator():

    def __init__(self, parent):
        NotImplemented

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