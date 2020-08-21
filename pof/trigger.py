"""

    Filename: trigger.py
    Description: Contains the code for implementing a trigger class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import numpy as np

"""

Examples - inspection - (initiated and outage) and (within any conditions)

    (AND State) AND (OR Condition)

Examples - Modificication - within all conditions

    (AND Condition)

Examples - Corrective Maintenance within any state

    (AND State)

Examples - Next Maintenance - (initiated and outage) and

    (AND State) AND (any condition)

    (AND State) AND (all condition)


"""

"""
States if not there return TRUE 

True or


"""


class Trigger():

    """
    formula
    """

    # Class constatns
    VALID_LOGIC = ['and', 'or']
    DEFAULT_LOGIC = 'and'

    def __init__(self, times=None, states=None, conditions=None, condition_logic=None, state_logic=None, overall_logic=None):

        # Triggers that can be used
        self.set_triggers(states=states, conditions=conditions, times=times)

        self.set_logic(condition_logic=condition_logic, state_logic=state_logic, overall_logic=overall_logic)

    # **************************** Instance Methods ****************************

    def set_logic(self, condition_logic=None, state_logic=None, overall_logic=None):
        """
        Set the logic for condition, state and overall
        """
        self._condition_logic = self._is_valid_logic(condition_logic)
        self._state_logic = self._is_valid_logic(state_logic)
        self._overall_logic = self._is_valid_logic(overall_logic)

    def set_triggers_all(self, triggers):

        self.set_triggers(states= triggers['states'], conditions = triggers['conditions'])

    def set_triggers(self, states = None, conditions=None, times=None):

        self.times = times if times is not None else dict()
        self.states = states if states is not None else dict()
        self.conditions = conditions if conditions is not None else dict()
        

    def check(self, timeline, t_start = 0, t_end = None):

        # Get start and end times
        if t_end is None:
            t_end = len(timeline['time'])
        else:
            t_end = min(len(timeline['time']), t_end)

        # Check for state triggers
        triggered = self.check_state(timeline, t_start, t_end)
        
        # Check for condition triggers
        triggered = triggered & self.check_condition(timeline, t_start, t_end)

        return triggered

    def check_state(self, timeline, t_start, t_end):

        # Initialise the trigger state
        if bool(self.states):

            # Create the the state logic
            if self._state_logic == 'or':
                triggered = np.full(t_end - t_start, False)
            
            elif self._state_logic == 'and':
                triggered = np.full(t_end - t_start, True)

            try:
                # Check the condition triggers have been met
                for state, target in self.states.items():

                    if self._state_logic == 'and':

                        triggered  = (triggered) & (timeline[state]==target)

                    elif self._state_logic == 'or':

                        triggered  = (triggered) | (timeline[state]==target)

            except KeyError:
                print ("%s not found" %(state))

        else:

            triggered = np.full(t_end - t_start, True)

        return triggered

    def check_condition(self, timeline, t_start, t_end):
        """
        Check if condition triggers have been met using a timeline
        """

        # Check if there are condition triggers
        if bool(self.conditions):

            # Create the start state
            if self._condition_logic == 'and':
                triggered = np.full(t_end - t_start, True)
            
            elif self._condition_logic == 'or':
                triggered = np.full(t_end - t_start, False)

            try:

                for condition, trigger in self.conditions.items():
                    
                    in_condition_window = (timeline[condition][t_start:t_end] >= trigger['lower']) & (timeline[condition][t_start:t_end] <= trigger['upper'])

                    if self._condition_logic == 'and':
                            triggered  = (triggered) & (in_condition_window)

                    elif self._condition_logic == 'or':
                            triggered  = (triggered) | (in_condition_window)

            except KeyError:
                print ("%s not found" %(condition))
        else:
            # No triggers to check so return true
            triggered = np.full(t_end - t_start, True)

        return triggered

    def check_value(self): # TODO placeholder concept for @Stephen Fisher

        return NotImplemented   
        

    # **************************** Instance Methods ****************************

    def _is_valid_logic(self, logic=None):
        """
        Check if logic is valid and return default if it is not valid
        """
        if logic not in Trigger.VALID_LOGIC:

            if logic is not None:
                print ("Only %s logic allowed" %(Trigger.VALID_LOGIC))

            logic = Trigger.DEFAULT_LOGIC

        return logic

if __name__ == "__main__":
    trigger = Trigger()
    print("Trigger - OK")