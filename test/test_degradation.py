

import unittest

from pof.degradation import Degradation

class TestDegradation(unittest.TestCase):

    def test_class_creation(self):
        deg = Degradation()

        self.assertTrue(True)
        

    def test_degradation_array(self):
        self.assertTrue(False)
        

    # Check the boundary cases
    
"""
    def test_consume_food_consumes_the_apple(self):
        c = Consumer()
        c.consume_food()
        self.assertTrue(c.apple.consumed,
                        "Expected apple to be consumed")

    def test_consume_food_cuts_the_food(self):
        c = Consumer()
        c.consume_food()
        self.assertTrue(c.apple.been_cut,
                        "Expected apple to be cut")

    def test_pick_food_always_selects_the_apple(self):
        c = Consumer()
        food = c.pick_food()
        self.assertEquals(c.apple, food,
                          "Expected apple to have been picked")"""


if __name__ == '__main__':
    unittest.main()