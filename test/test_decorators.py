import unittest

import testconfig
from pof.decorators import check_arg_positive, coerce_arg_type, check_arg_type


class TestDecorators(unittest.TestCase):
    def test_check_arg_positive_errors(self):

        params

        def func(a, b, c=-3, *args, **kwargs):
            return (a, b, c)

    def test_chain(self):
        @check_arg_positive("c")
        @coerce_arg_type
        def func(a, b, c=-3, *args, **kwargs):
            return (a, b, c)

        func(1, 2, 3)

    def test_check_arg_positive(self):
        @check_arg_type
        @check_arg_type
        @check_arg_positive("c")
        def func(a, b, c: float, *args, **kwargs):
            return (a, b, c)

        func(1, 2, "3")
