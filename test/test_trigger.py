
"""

    Filename: test_trigger.py
    Description: Contains the code for testing the trigger class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
import numpy as np

from pof.trigger import Trigger

class TestTrigger(unittest.TestCase):

    def get_test_triggers(self, triggers_to_keep):
        """
        A function to make it easier to get test data
        """

        triggers = dict(
            conditions = {k: v for (k, v) in self.cond_options.items() if k in triggers_to_keep},
            states = {k: v for (k, v) in self.state_options.items() if k in triggers_to_keep},
        )

        return triggers

    def setUp(self):

        self.timeline = dict(
            time = np.linspace(0, 100, 101),
            condition_1 = np.linspace(100, 0 , 101),
            condition_2 = np.linspace(100, 0 , 101),
            state_1 = np.concatenate((np.full(10,False), np.full(81, True), np.full(10, False))),
            state_2 = np.concatenate((np.full(20,False), np.full(61, True), np.full(20, False))),
        )

        self.cond_options = dict(
                condition_1 = dict(
                    lower = 50,
                    upper = 100,
                ),
                condition_2 = dict(
                    lower = 70,
                    upper = 90, 
                ),
        
        )
        self.state_options = dict(
                state_1 = True,
                state_2 = True,
        )
    

    def test_class_imports_correctly(self):
        self.assertTrue(True)

    def test_class_instantiate(self):
        t = Trigger()
        self.assertTrue(True)

    # *************** Test Set Logic ***********************

    def test_set_logic_valid(self):
        t = Trigger()

        for test_logic in t.VALID_LOGIC:

            t.set_logic(condition_logic=test_logic, state_logic=test_logic, overall_logic=test_logic)

            self.assertEqual(test_logic, t._condition_logic)
            self.assertEqual(test_logic, t._state_logic)
            self.assertEqual(test_logic, t._overall_logic)

    def test_set_logic_none(self):
        t = Trigger()
        test_logic = None

        t.set_logic(condition_logic=test_logic, state_logic=test_logic, overall_logic=test_logic)

        self.assertEqual(t.DEFAULT_LOGIC, t._condition_logic)
        self.assertEqual(t.DEFAULT_LOGIC, t._state_logic)
        self.assertEqual(t.DEFAULT_LOGIC, t._overall_logic)

    def test_set_logic_invalid(self):
        t = Trigger()
        test_logic = str(t.DEFAULT_LOGIC).join('_invalid')

        t.set_logic(condition_logic=test_logic, state_logic=test_logic, overall_logic=test_logic)

        self.assertEqual(t.DEFAULT_LOGIC, t._condition_logic)
        self.assertEqual(t.DEFAULT_LOGIC, t._state_logic)
        self.assertEqual(t.DEFAULT_LOGIC, t._overall_logic)
    
    # *************** Test Check Condition ***********************

    def test_check_condition_and_one_condition(self):

        expected = np.concatenate((
            np.full(51,True),
            np.full(50, False)
        ))

        triggers = self.get_test_triggers(['condition_1'])

        t = Trigger()
        t.set_logic(condition_logic='and')
        t.set_triggers_all(triggers)
        output = t.check_condition(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_condition_and_two_condition(self):

        expected = np.concatenate((
            np.full(10, False),
            np.full(21,True),
            np.full(70, False)
        ))

        triggers = self.get_test_triggers(['condition_1', 'condition_2'])

        t = Trigger()
        t.set_logic(condition_logic='and')
        t.set_triggers_all(triggers)
        output = t.check_condition(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_condition_or_one_condition(self):

        expected = np.concatenate((
            np.full(51,True),
            np.full(50, False)
        ))

        triggers = self.get_test_triggers(['condition_1'])

        t = Trigger()
        t.set_logic(condition_logic='or')
        t.set_triggers_all(triggers)
        output = t.check_condition(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_condition_or_two_condition(self):

        expected = np.concatenate((
            np.full(51,True),
            np.full(50, False)
        ))

        triggers = self.get_test_triggers(['condition_1', 'condition_2'])

        t = Trigger()
        t.set_logic(condition_logic='or')
        t.set_triggers_all(triggers)
        output = t.check_condition(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    # *************** Test Check State ***********************

    def test_check_state_and_one_state(self):

        expected = np.concatenate((
            np.full(10,False),
            np.full(81, True),
            np.full(10, False)
        ))

        triggers = self.get_test_triggers(['state_1'])

        t = Trigger()
        t.set_logic(state_logic='and')
        t.set_triggers_all(triggers)
        output = t.check_state(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_state_and_two_state(self):

        expected = np.concatenate((
            np.full(20,False),
            np.full(61, True),
            np.full(20, False)
        ))

        triggers = self.get_test_triggers(['state_1', 'state_2'])

        t = Trigger()
        t.set_logic(state_logic='and')
        t.set_triggers_all(triggers)
        output = t.check_state(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_state_or_one_state(self):

        expected = np.concatenate((
            np.full(10, False),
            np.full(81, True),
            np.full(10, False)
        ))

        triggers = self.get_test_triggers(['state_1'])

        t = Trigger()
        t.set_logic(state_logic='or')
        t.set_triggers_all(triggers)
        output = t.check_state(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

    def test_check_state_or_two_state(self):

        expected = np.concatenate((
            np.full(10,False),
            np.full(81, True),
            np.full(10, False)
        ))

        triggers = self.get_test_triggers(['state_1', 'state_2'])

        t = Trigger()
        t.set_logic(state_logic='or')
        t.set_triggers_all(triggers)
        output = t.check_state(self.timeline, t_start=0, t_end=101)

        np.testing.assert_array_equal(expected, output)

# *************** Test Check ***********************

    def test_check_overall_and_state_and_condition_and(self):

        expected = np.concatenate((
            np.full(20,False),
            np.full(11, True),
            np.full(70, False)
        ))

        triggers = self.get_test_triggers(['state_1', 'state_2', 'condition_1', 'condition_2'])

        t = Trigger()
        t.set_logic(state_logic='and', condition_logic='and', overall_logic='and')
        t.set_triggers_all(triggers)
        output = t.check(self.timeline)

        np.testing.assert_array_equal(expected, output)

    def test_check_overall_or_state_and_condition_and(self):

        expected = np.concatenate((
            np.full(20,False),
            np.full(11, True),
            np.full(70, False),
        ))

        triggers = self.get_test_triggers(['state_1', 'state_2', 'condition_1', 'condition_2'])

        t = Trigger()
        t.set_logic(state_logic='and', condition_logic='and', overall_logic='and')
        t.set_triggers_all(triggers)
        output = t.check(self.timeline)

        np.testing.assert_array_equal(expected, output)

if __name__ == '__main__':
    unittest.main()
