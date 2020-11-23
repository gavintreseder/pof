import unittest

import testconfig
from pof.decorators import check_arg_positive, coerce_arg_type, check_arg_type


# given, should, actual, expected


class TestDecorators(unittest.TestCase):


    def gen_arg_inputs(*args, **kwargs):
        return arg


    @check_arg_positive("b", "c")
    def arg_positive(a, b=2, c=-3, *args, **kwargs):
        return (a, b, c)


    def test_check_arg_positive(self):

        # args, kwargs
        param_inputs = [
            ((-1,1,1,-1), {})
            ((), {'a':-1, 'b': 1, 'c':1, 'd': -1}),
            ((),{'c':1}),
        ]

        for args, kwargs in param_inputs:

            given = "positive inputs"
            should = "execute the function without error"
            actual = self.arg_positive(*args, **kwargs)
            expected = (*args, **kwargs)

            self.assertEqual(actual, expected, msg=f"Given {given}: should {should}")

    def test_check_arg_positive_with_errors(self):

        # args, kwargs
        param_inputs = [
            ((1,-1,-1,1), {})
            ((), {'a':1, 'b': -1, 'c':-1, 'd': 1}),
            ((),{}),
        ]

        for args, kwargs in param_inputs:

            with self.assertraises():
                given = "negative inputs"
                should = "raise an error"
                actual = self.arg_positive(*args, **kwargs)
                expected = (*args, **kwargs)

                self.assertEqual(actual, expected, msg=f"Given {given}: should {should}")


    def test_chain(self):
        @coerce_arg_type
        @check_arg_positive("c")
        def func(a, b, c: int = 3, *args, **kwargs):
            return (a, b, c)

        func(1, 2, "3")
