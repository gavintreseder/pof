import unittest

import testconfig
from pof.decorators import check_arg_positive, coerce_arg_type, check_arg_type


class TestDecorators(unittest.TestCase):
    def test_check_arg_positive_errors(self):

        params

        def func(a, b, c=-3, *args, **kwargs):
            return (a, b, c)

    def test_chain(self):
        @coerce_arg_type
        @check_arg_positive("c")
        def func(a, b, c: int = 3, *args, **kwargs):
            return (a, b, c)

        func(1, 2, "3")
